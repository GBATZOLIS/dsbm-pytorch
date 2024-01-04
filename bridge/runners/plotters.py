import time
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import torchvision.utils as vutils
import hydra
import glob
import itertools

from ..data.utils import save_image, to_uint8_tensor, normalize_tensor
from ..data.metrics import PSNR, SSIM, FID  # , LPIPS
from PIL import Image
# matplotlib.use('Agg')
from tqdm import tqdm

DPI = 200

def make_gif(plot_paths, output_directory='./gif', gif_name='gif'):
    frames = [Image.open(fn) for fn in plot_paths]

    frames[0].save(os.path.join(output_directory, f'{gif_name}.gif'),
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100,
                   loop=0)



class Plotter(object):
    def __init__(self, ipf, args, im_dir = './im', gif_dir='./gif', is_SB=True, test=True):
        self.ipf = ipf
        self.args = args
        self.plot_level = self.args.plot_level

        self.dataset = self.args.data.dataset
        self.num_steps = self.ipf.test_num_steps #default number of integration steps

        if self.ipf.accelerator.is_main_process:
            os.makedirs(im_dir, exist_ok=True)
            os.makedirs(gif_dir, exist_ok=True)

        self.im_dir = im_dir
        self.gif_dir = gif_dir
        self.is_SB = is_SB
        self.test = test

        self.metrics_dict = {}

    def __call__(self, i, n, fb, sampler='sde', datasets='all', ode_method='odeint', \
                    calc_energy=False, num_steps='default', \
                    return_tensors=False, x_start=None):
        assert sampler in ['sde', 'ode']
        out = {}
        self.step = self.ipf.compute_current_step(i, n)

        if self.is_SB:
            cache_filepath_npy = sorted(glob.glob(os.path.join(self.ipf.cache_dir, f"cache_{fb}_{n:03}.npy")))
        
        if datasets == 'all':
            datasets = ['train', 'test']

        if self.ipf.accelerator.is_main_process:
            if self.is_SB:
                out['fb'] = fb
                out['ipf'] = n
                out['T'] = self.ipf.T

        for dl_name, dl in self.ipf.save_dls_dict.items():
            if dl_name not in datasets:
                continue

            
            #use_cache = ((dl_name == "train") and (sampler == 'sde') and (self.step >= self.ipf.compute_current_step(0, n+1)) and 
            #             (len(cache_filepath_npy) > 0) and (self.ipf.cache_num_steps == self.num_steps))
            use_cache = False
            '''
            x_start, y_start, x_tot, x_init, mean_final, var_final, metric_results = \
                self.generate_sequence_joint(dl, i, n, fb, dl_name=dl_name, sampler=sampler, num_steps=num_steps,
                                                ode_method=ode_method, calc_energy=False)

            if self.ipf.accelerator.is_main_process:
                self.plot_sequence_joint(x_start[:self.args.plot_npar], y_start[:self.args.plot_npar],
                                         x_tot[:, :self.args.plot_npar], x_init[:self.args.plot_npar],
                                         self.dataset, i, n, fb, dl_name=dl_name, sampler=sampler,
                                         mean_final=mean_final, var_final=var_final, num_steps=num_steps)
            '''

            if use_cache and not self.ipf.cdsb and self.is_SB:
                print("Using cached data for training set evaluation")
                fp = np.load(cache_filepath_npy[0], mmap_mode="r")
                all_x = torch.from_numpy(fp[:self.ipf.test_npar])
                if fb == 'f':
                    x_start, x_last = all_x[:, 0], all_x[:, 1]
                else:
                    x_start, x_last = all_x[:, 1], all_x[:, 0]
                y_start, x_init = [], []

            else:
                fbs = ['b', 'f'] if self.args.space == 'latent' else ['b']
                if fb in fbs and dl_name == 'train' and self.test:
                    generate_npar = self.ipf.test_npar
                else:
                    generate_npar = min(self.ipf.test_npar, self.ipf.plot_npar)
                
                
                start_time = time.time()
                x_start, y_start, x_tot, x_init, mean_final, var_final, metric_results = \
                    self.generate_sequence_joint(dl, i, n, fb, dl_name=dl_name, sampler=sampler, 
                                                generate_npar=generate_npar, full_traj=False, num_steps=num_steps,
                                                ode_method=ode_method, calc_energy=calc_energy, x_start=x_start)
                
                end_time = time.time()
                print(f"Generating test dataset with {num_steps} integration steps takes {end_time - start_time} seconds to execute")
                

                if self.ipf.accelerator.is_main_process:
                    self.plot_sequence_joint(x_start[:self.args.plot_npar], y_start[:self.args.plot_npar],
                                            x_tot[:, :self.args.plot_npar], x_init[:self.args.plot_npar],
                                            self.dataset, i, n, fb, dl_name=dl_name, sampler=sampler,
                                            mean_final=mean_final, var_final=var_final, num_steps=num_steps)

                x_last = x_tot[-1]
                print(f'size of test dataset: {x_last.size()}')

            x_tot = None

            if generate_npar == self.ipf.test_npar and self.test:
                test_results = self.test_joint(x_start[:generate_npar], y_start[:generate_npar], 
                                            x_last[:generate_npar], x_init[:generate_npar],
                                            i, n, fb, dl_name=dl_name, sampler=sampler,
                                            mean_final=mean_final, var_final=var_final)
                test_results = {self.prefix_fn(dl_name, sampler, num_steps) + k: v for k, v in test_results.items()}
                out.update(test_results)

            metric_results = {self.prefix_fn(dl_name, sampler, num_steps) + k: v for k, v in metric_results.items()}
            out.update(metric_results)
            

        torch.cuda.empty_cache()
        if return_tensors:
            images = {'x_start':x_start,
                      'y_start':y_start,
                      'x_last':x_last,
                      'x_init':x_init}
            return out, images
        else:
            return out


    def prefix_fn(self, dl_name, sampler, num_steps='default'):
        assert sampler in ['sde', 'ode']
        if sampler == 'sde':
            if num_steps == 'default':
                num_steps = self.num_steps
            return dl_name + '/' + str(num_steps) + '/'
        else:
            return dl_name + '/ode/'

    def process_energy_steps(self, all_tsteps, all_energies):
        # Assume all_tsteps is a list of tensors
        # Check if all tsteps tensors are the same
        if all(torch.equal(tsteps, all_tsteps[0]) for tsteps in all_tsteps[1:]):
            # Calculate the mean energy over all batches for each tstep
            # Move each tensor in the tuple to CPU before taking the mean
            mean_energies = torch.tensor([torch.mean(torch.stack(energies)) for energies in zip(*all_energies)])
            return all_tsteps[0], mean_energies
        else:
            # Concatenate and sort by tsteps, average energies for duplicated tsteps
            combined = sorted(zip(itertools.chain(*all_tsteps), itertools.chain(*all_energies)))
            processed_tsteps, processed_energies = [], []
            for key, group in itertools.groupby(combined, lambda x: x[0]):
                energies = [g[1] for g in group]
                # Here you have a list of tensors, stack them and then move to CPU
                stacked_energies = torch.stack(energies)
                processed_tsteps.append(key)
                processed_energies.append(torch.mean(stacked_energies).item())  # .item() to get a plain number
            processed_energies = torch.tensor(processed_energies)
            return processed_tsteps, processed_energies


    def generate_sequence_joint(self, dl, i, n, fb, dl_name='train', sampler='sde', generate_npar=None, 
                            full_traj=True, num_steps='default', ode_method='odeint', calc_energy=False, x_start=None):
        iter_dl = iter(dl)

        all_batch_x = []
        all_batch_y = []
        all_x_tot = []
        all_init_batch_x = []
        all_mean_final = []
        all_var_final = []
        all_tsteps = []
        all_energies = []

        times = []
        nfes = []
        metric_results = {}

        if generate_npar is None:
            generate_npar = self.ipf.plot_npar
        
        if num_steps == 'default':
            num_steps = self.num_steps

        iters = 0
        while iters * self.ipf.test_batch_size < generate_npar:
            try:
                start = time.time()

                init_batch_x, batch_y, final_batch_x, mean_final, var_final = self.ipf.sample_batch(iter_dl, self.ipf.save_final_dl_repeat) #, phase='test')

                with torch.no_grad():
                    if fb == 'f':
                        if x_start is None:
                            batch_x = init_batch_x
                        else:
                            index_start = iters * self.ipf.test_batch_size
                            index_end = (iters+1) * self.ipf.test_batch_size
                            batch_x = x_start[index_start:index_end]

                        if sampler == 'ode':
                            x_tot, nfe = self.ipf.forward_sample_ode(batch_x, batch_y, permute=False)
                        else:
                            x_tot, nfe = self.ipf.forward_sample(batch_x, batch_y, permute=False, num_steps=num_steps)
                        x_last_true = final_batch_x

                    else:
                        if x_start is None:
                            batch_x = final_batch_x
                        else:
                            index_start = iters * self.ipf.test_batch_size
                            index_end = (iters+1) * self.ipf.test_batch_size
                            batch_x = x_start[index_start:index_end]

                        if sampler == 'ode':
                            '''
                            x_tot, nfe, tsteps, energies = self.ipf.backward_sample_ode_under_testing(batch_x, batch_y, permute=False, 
                                                                                                    num_steps=num_steps,
                                                                                                    method=ode_method, calc_energy=calc_energy)
                            all_tsteps.append(tsteps)
                            all_energies.append(energies)
                            '''
                            x_tot, nfe = self.ipf.backward_sample_ode(batch_x, batch_y, permute=False, num_steps=num_steps)
                        else:
                            x_tot, nfe, tsteps, energies = self.ipf.backward_sample(batch_x, batch_y, permute=False, num_steps=num_steps, calc_energy=calc_energy)  # var_final=var_final, 
                            all_tsteps.append(tsteps)
                            all_energies.append(energies)

                        x_last_true = init_batch_x

                    stop = time.time()
                    times.append(stop - start)

                    if iters==0:
                        batchtime = stop - start
                        minutes, seconds = divmod(batchtime, 60)
                        print(f'It takes {minutes} minutes and {seconds} seconds to generate: {batch_x.size(0)} images')

                        test_batch_size = batch_x.size(0)
                        expected_time_seconds = generate_npar / test_batch_size * batchtime
                        expected_time_minutes, expected_time_remaining_seconds = divmod(expected_time_seconds, 60)
                        expected_time_hours, expected_time_remaining_minutes = divmod(expected_time_minutes, 60)
                        print(f'It should take about: {int(expected_time_hours)} hours, {int(expected_time_remaining_minutes)} minutes, and {int(expected_time_remaining_seconds)} seconds to generate the dataset.')


                    nfes.append(nfe)

                    gather_batch_x = self.ipf.accelerator.gather(batch_x)
                    if self.ipf.cdsb:
                        gather_batch_y = self.ipf.accelerator.gather(batch_y)
                    gather_init_batch_x = self.ipf.accelerator.gather(x_last_true)

                    if not full_traj:
                        x_tot = x_tot[:, -1:].contiguous()
                        gather_x_tot = self.ipf.accelerator.gather(x_tot)
                    else:
                        gather_x_tot = x_tot

                    all_batch_x.append(gather_batch_x.cpu())
                    if self.ipf.cdsb:
                        all_batch_y.append(gather_batch_y.cpu())
                    all_x_tot.append(gather_x_tot.cpu())
                    all_init_batch_x.append(gather_init_batch_x.cpu())

                    iters = iters + 1

            except StopIteration:
                break

        all_batch_x = torch.cat(all_batch_x, dim=0)
        if self.ipf.cdsb:
            all_batch_y = torch.cat(all_batch_y, dim=0)
        all_x_tot = torch.cat(all_x_tot, dim=0)
        all_init_batch_x = torch.cat(all_init_batch_x, dim=0)

        shape_len = len(all_x_tot.shape)
        all_x_tot = all_x_tot.permute(1, 0, *list(range(2, shape_len)))

        all_mean_final = self.ipf.mean_final.cpu()
        all_var_final = self.ipf.var_final.cpu()

        metric_results['nfe'] = np.mean(nfes)
        metric_results['batch_sample_time'] = np.mean(times)

        if calc_energy:
            processed_tsteps, processed_energies = self.process_energy_steps(all_tsteps, all_energies)
            metric_results['tsteps'] = processed_tsteps
            metric_results['energies'] = processed_energies
            metric_results['energy'] = torch.mean(processed_energies).item()

        return all_batch_x, all_batch_y, all_x_tot, all_init_batch_x, all_mean_final, all_var_final, metric_results

    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', sampler='sde', freq=None,
                            mean_final=None, var_final=None, num_steps='default'):
        pass

    def test_joint(self, x_start, y_start, x_last, x_init, i, n, fb, dl_name='train', sampler='sde', mean_final=None, var_final=None):
        out = {}
        metric_results = {}

        x_var_last = torch.var(x_last, dim=0).mean().item()
        x_var_start = torch.var(x_start, dim=0).mean().item()
        x_mean_last = torch.mean(x_last).item()
        x_mean_start = torch.mean(x_start).item()

        x_mse_start_last = torch.mean((x_start - x_last) ** 2).item()

        '''
        out = {'x_mean_start': x_mean_start, 'x_var_start': x_var_start,
                'x_mean_last': x_mean_last, 'x_var_last': x_var_last, 
                'x_mse_start_last': x_mse_start_last}
        '''

        if mean_final is not None:
            x_mse_last = torch.mean((x_last - mean_final) ** 2).item()
            x_mse_start = torch.mean((x_start - mean_final) ** 2).item()
            #out.update({"x_mse_start": x_mse_start, "x_mse_last": x_mse_last})

        if fb == 'b' or (self.ipf.transfer and dl_name == 'train'):
            dl_x_start = self.ipf.build_dataloader(x_start, batch_size=self.ipf.test_batch_size, shuffle=False, repeat=False)
            dl_x_start = iter(dl_x_start)
            if self.ipf.cdsb and len(y_start) > 0:
                dl_y_start = self.ipf.build_dataloader(y_start, batch_size=self.ipf.test_batch_size, shuffle=False, repeat=False)
                dl_y_start = iter(dl_y_start)
            else:
                dl_y_start = None
            dl_x_last = self.ipf.build_dataloader(x_last, batch_size=self.ipf.test_batch_size, shuffle=False, repeat=False)
            dl_x_last = iter(dl_x_last)
            if len(x_init) > 0:
                dl_x_init = self.ipf.build_dataloader(x_init, batch_size=self.ipf.test_batch_size, shuffle=False, repeat=False)
                dl_x_init = iter(dl_x_init)
            else:
                dl_x_init = None
            dl_x_last_true = self.ipf.save_dls_dict[dl_name] if fb == 'b' else self.ipf.save_final_dl
            dl_x_last_true = iter(dl_x_last_true)



            for metric_name, metric in self.metrics_dict.items():
                metric.reset()
                
            iters = 0
            while iters * self.ipf.test_batch_size < self.ipf.test_npar: #we generate test_npar (usually 10K) samples for FID evaluation
                try:
                    x_start, x_last = next(dl_x_start), next(dl_x_last)
                    if dl_y_start is not None:
                        y_start = next(dl_y_start)
                    else:
                        y_start = None
                    if dl_x_init is not None:
                        x_init = next(dl_x_init)
                    else:
                        x_init = None
                    x_last_true, _ = next(dl_x_last_true)

                    self.plot_and_record_batch_joint(x_start, y_start, x_last, x_init, x_last_true, iters, i, n, fb, dl_name=dl_name, sampler=sampler)

                    iters = iters + 1
                
                except StopIteration:
                    break

            if iters > 0:
                for metric_name, metric in self.metrics_dict.items():
                    # Timing metric computation
                    metric_start_time = time.time()
                    metric_result = metric.compute()
                    print(f"Metric Computation Time: {time.time() - metric_start_time} seconds")

                    if self.ipf.accelerator.is_main_process:
                        metric_results[metric_name] = metric_result
                    metric.reset()
        

        out.update(metric_results)
        #out.update({'test_npar': self.ipf.test_npar})
        return out

    def plot_and_record_batch_joint(self, x_start, y_start, x_last, x_init, x_last_true, iters, i, n, fb, dl_name='train', sampler='sde'):
        pass

    def save_image(self, tensor, name, dir, **kwargs):
        return []

class LatentPlotter(Plotter):
    def __init__(self, ipf, args, im_dir = './im', gif_dir='./gif', is_SB=True, test=True):
        super().__init__(ipf, args, im_dir=im_dir, gif_dir=gif_dir, is_SB=is_SB, test=test)
        self.num_plots_grid = 100
        self.fid = FID().to(self.ipf.device)
        self.fid_start = FID().to(self.ipf.device) #baseline

    def flatten_tensor_if_needed(self, tensor):
        if tensor.dim() > 2:  # Check if tensor is in a format more complex than (batch, d)
            return tensor.view(tensor.size(0), -1)  # Flatten to (batch, -1)
        return tensor

    def save_image(self, tensor, name, dir, **kwargs):
        fp = os.path.join(dir, f'{name}.png')
        save_image(tensor[:self.num_plots_grid], fp, nrow=10)
        return [fp]
        
    def matrix_sqrt(self, mat):
        U, S, V = torch.svd(mat)
        return U @ torch.diag(torch.sqrt(S)) @ V.t()

    def compute_wasserstein_distance(self, mean1, cov1, mean2, cov2):
        diff_mean = mean1 - mean2
        sqrt_cov1 = self.matrix_sqrt(cov1)
        #sqrt_cov2 = self.matrix_sqrt(cov2)
        
        # Squared Euclidean distance between the means
        term1 = torch.norm(diff_mean, p=2)**2
        
        # The trace term in the Wasserstein distance formula
        term2 = torch.trace(cov1 + cov2 - 2 * self.matrix_sqrt(sqrt_cov1 @ cov2 @ sqrt_cov1))
        
        # Compute the Wasserstein distance
        w2_distance = term1 + term2
        return w2_distance.item()

    def decode_with_dataloader(self, tensor, batch_size):
        dataloader = self.ipf.build_dataloader(tensor, batch_size=batch_size, shuffle=False, drop_last=False, repeat=False)
        decoded_batches = []

        for batch in dataloader:
            decoded_batch, _ = self.ipf.vae_model.decode(batch.to(self.ipf.device))
            decoded_batches.append(decoded_batch.cpu())

        return torch.cat(decoded_batches, dim=0)

    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', sampler='sde', freq=None,
                            mean_final=None, var_final=None, num_steps='default'):
        if fb == 'f':
            pass
        elif fb == 'b':
            current_batch_size = self.num_plots_grid
            decoded_successfully = False

            while not decoded_successfully and current_batch_size > 0:
                try:
                    x_last = self.decode_with_dataloader(x_tot[:, :self.num_plots_grid][-1], current_batch_size)
                    x_start = self.decode_with_dataloader(x_start[:self.num_plots_grid], current_batch_size)
                    x_init = self.decode_with_dataloader(x_init[:self.num_plots_grid], current_batch_size)
                    decoded_successfully = True
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"Caught OOM error with batch size {current_batch_size}, reducing by half.")
                        current_batch_size //= 2  # Reduce the batch size by half
                    else:
                        raise e  # Re-raise if it's not an OOM error

            if not decoded_successfully:
                raise RuntimeError("Unable to decode: continuous OOM errors.")

            name = str(i) + '_' + fb + '_' + str(n)
            im_dir = os.path.join(self.im_dir, name, self.prefix_fn(dl_name, sampler, num_steps))
            os.makedirs(im_dir, exist_ok=True)

            filename_grid = 'im_grid_start'
            print(x_start.size())
            filepath_grid_list = self.save_image(x_start.cpu(), filename_grid, im_dir)
            print(filepath_grid_list)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler, num_steps) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            filename_grid = 'im_grid_last'
            print(x_last.size())
            filepath_grid_list = self.save_image(x_last.cpu(), filename_grid, im_dir)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler, num_steps) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            filename_grid = 'im_grid_data_x'
            print(x_init.size())
            filepath_grid_list = self.save_image(x_init.cpu(), filename_grid, im_dir)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler, num_steps) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

    def test_joint(self, x_start, y_start, x_last, x_init, i, n, fb, dl_name='train', sampler='sde', mean_final=None, var_final=None):
        out = {}

        # Create flattened copies for Wasserstein distance calculation
        flat_x_start = self.flatten_tensor_if_needed(x_start)
        flat_x_last = self.flatten_tensor_if_needed(x_last)

        # Calculate means and covariances on the flattened tensors
        mean_flat_x_start = torch.mean(flat_x_start, dim=0)
        cov_flat_x_start = torch.cov(flat_x_start.t())
        mean_flat_x_last = torch.mean(flat_x_last, dim=0)
        cov_flat_x_last = torch.cov(flat_x_last.t())

        if fb == 'b':
            # Wasserstein-2 distance between x_last and x_init
            flat_x_init = self.flatten_tensor_if_needed(x_init)
            mean_flat_x_init = torch.mean(flat_x_init, dim=0)
            cov_flat_x_init = torch.cov(flat_x_init.t())

            #compute the FID scores
            dl_x_init = self.ipf.build_dataloader(x_init, batch_size=self.ipf.test_batch_size, shuffle=False, repeat=False)
            dl_x_init = iter(dl_x_init)
            dl_x_last = self.ipf.build_dataloader(x_last, batch_size=self.ipf.test_batch_size, shuffle=False, repeat=False)
            dl_x_last = iter(dl_x_last)
            dl_x_start = self.ipf.build_dataloader(x_start, batch_size=self.ipf.test_batch_size, shuffle=False, repeat=False)
            dl_x_start = iter(dl_x_start)

            # Calculate the total number of iterations needed
            total_iters = (self.ipf.test_npar + self.ipf.test_batch_size - 1) // self.ipf.test_batch_size

            print('Decoding of the latents and calculation of FID scores.')
            with tqdm(total=total_iters, desc="Computing FID") as pbar:
                iters = 0
                while iters * self.ipf.test_batch_size < self.ipf.test_npar:
                    try:
                        x_init, x_last, x_start = next(dl_x_init), next(dl_x_last), next(dl_x_start)
                        #x_init:ground truth latent
                        #x_last:generated latent

                        decoded_true, _ = self.ipf.vae_model.decode(x_init)
                        decoded_last, _ = self.ipf.vae_model.decode(x_last)
                        decoded_start, _ = self.ipf.vae_model.decode(x_start)

                        uint8_x_last_true = to_uint8_tensor(decoded_true)
                        uint8_x_last = to_uint8_tensor(decoded_last)
                        self.fid.update(uint8_x_last, uint8_x_last_true)

                        uint8_x_start = to_uint8_tensor(decoded_start)
                        self.fid_start.update(uint8_x_start, uint8_x_last_true)

                        # Update the progress bar
                        pbar.update(1)
                        iters += 1

                    except StopIteration:
                        break
            
            fid_result = self.fid.compute()
            fid_start_result = self.fid_start.compute()

            out['fid'] = fid_result
            out['fid_start'] = fid_start_result

            self.fid.reset()
            self.fid_start.reset()

        elif fb == 'f':
            # Wasserstein-2 distance between x_last and N(0, I)
            latent_dim = flat_x_last.size(1)
            mean_flat_x_init = torch.zeros(latent_dim)
            cov_flat_x_init = torch.eye(latent_dim)

        # Wasserstein-2 distances calculation common for both 'b' and 'f' cases
        out['wasserstein_distance_start'] = self.compute_wasserstein_distance(mean_flat_x_start, cov_flat_x_start, mean_flat_x_init, cov_flat_x_init)
        out['wasserstein_distance'] = self.compute_wasserstein_distance(mean_flat_x_last, cov_flat_x_last, mean_flat_x_init, cov_flat_x_init)
        
        return out

class ImPlotter(Plotter):

    def __init__(self, ipf, args, im_dir = './im', gif_dir='./gif', fid_feature=2048, is_SB=True, test=True):
        super().__init__(ipf, args, im_dir=im_dir, gif_dir=gif_dir, is_SB=is_SB, test=test)
        self.num_plots_grid = 100

        if fid_feature == 2048:
            fid_metric_name = "fid"
            self.metrics_dict = {fid_metric_name: FID().to(self.ipf.device)}
        elif fid_feature in [64, 192, 768]:
            fid_metric_name = "fid_%d" % fid_feature
            self.metrics_dict = {fid_metric_name: FID(feature=fid_feature).to(self.ipf.device)}
        else:
            raise NotImplementedError(f'fid_feature {fid_feature} is not supported by the FID torchmetric.')
        
        '''
        if self.dataset == "CIFAR10":
            data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)
            root = os.path.join(data_dir, 'cifar10')
            fid_stats = torch.load(os.path.join(root, 'fid_stats.pt'))
            self.metrics_dict[fid_metric_name].real_features_sum = fid_stats["real_features_sum"].to(self.ipf.device)
            self.metrics_dict[fid_metric_name].real_features_cov_sum = fid_stats["real_features_cov_sum"].to(self.ipf.device)
            self.metrics_dict[fid_metric_name].real_features_num_samples = fid_stats["real_features_num_samples"].to(self.ipf.device)
            self.metrics_dict[fid_metric_name].reset_real_features = False
        '''
        
    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', sampler='sde', freq=None,
                            mean_final=None, var_final=None, num_steps='default'):
        super().plot_sequence_joint(x_start, y_start, x_tot, x_init, data, i, n, fb, freq=freq, dl_name=dl_name, sampler=sampler,
                                    mean_final=mean_final, var_final=var_final, num_steps=num_steps)
        #num_steps = x_tot.shape[0]

        if freq is None:
            f_num_steps = x_tot.shape[0]
            freq = f_num_steps // min(f_num_steps, 50)

        if self.plot_level >= 1:
            x_tot_grid = x_tot[:, :self.num_plots_grid]
            name = str(i) + '_' + fb + '_' + str(n)
            im_dir = os.path.join(self.im_dir, name, self.prefix_fn(dl_name, sampler, num_steps))
            gif_dir = os.path.join(self.gif_dir, self.prefix_fn(dl_name, sampler, num_steps))

            os.makedirs(im_dir, exist_ok=True)
            os.makedirs(gif_dir, exist_ok=True)

            filename_grid = 'im_grid_start'
            filepath_grid_list = self.save_image(x_start[:self.num_plots_grid], filename_grid, im_dir)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler, num_steps) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            filename_grid = 'im_grid_last'
            filepath_grid_list = self.save_image(x_tot_grid[-1], filename_grid, im_dir)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler, num_steps) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            filename_grid = 'im_grid_data_x'
            filepath_grid_list = self.save_image(x_init[:self.num_plots_grid], filename_grid, im_dir)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler, num_steps) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            if self.plot_level >= 2:
                plot_paths = []
                x_start_tot_grid = torch.cat([x_start[:self.num_plots_grid].unsqueeze(0), x_tot_grid], dim=0)
                for k in range(num_steps+1):
                    if k % freq == 0 or k == num_steps:
                        # save png
                        filename_grid = 'im_grid_{0}'.format(k)
                        filepath_grid_list = self.save_image(x_start_tot_grid[k], filename_grid, im_dir)
                        plot_paths.append(filepath_grid_list[0])

                make_gif(plot_paths, output_directory=gif_dir, gif_name=name+'_im_grid')

    def plot_and_record_batch_joint(self, x_start, y_start, x_last, x_init, x_last_true, iters, i, n, fb, dl_name='train', sampler='sde'):
        if fb == 'b' or self.ipf.transfer:
            uint8_x_last_true = to_uint8_tensor(x_last_true)
            uint8_x_last = to_uint8_tensor(x_last)

            for metric in self.metrics_dict.values():
                metric.update(uint8_x_last, uint8_x_last_true)

            # if self.plot_level >= 3:
            #     name = str(i) + '_' + fb + '_' + str(n)
            #     im_dir = os.path.join(self.im_dir, name, dl_name)
            #     im_dir = os.path.join(im_dir, "im/")
            #     os.makedirs(im_dir, exist_ok=True)

            #     for k in range(x_last.shape[0]):
            #         plt.clf()
            #         file_idx = iters * self.ipf.test_batch_size + self.ipf.accelerator.process_index * self.ipf.test_batch_size // self.ipf.accelerator.num_processes + k
            #         filename_png = os.path.join(im_dir, '{:05}.png'.format(file_idx))
            #         assert not os.path.isfile(filename_png)
            #         save_image(x_last[k], filename_png)

    def save_image(self, tensor, name, dir, **kwargs):
        fp = os.path.join(dir, f'{name}.png')
        save_image(tensor[:self.num_plots_grid], fp, nrow=10)
        return [fp]


class DownscalerPlotter(Plotter):

    def __init__(self, ipf, args, im_dir = './im', gif_dir='./gif', is_SB=True, test=True):
        super().__init__(ipf, args, im_dir=im_dir, gif_dir=gif_dir, is_SB=is_SB, test=test)
        self.num_plots_grid = 16
        assert self.ipf.cdsb

    def plot_sequence_joint(self, x_start, y_start, x_tot, x_init, data, i, n, fb, dl_name='train', sampler='sde', freq=None,
                            mean_final=None, var_final=None):
        super().plot_sequence_joint(x_start, y_start, x_tot, x_init, data, i, n, fb, freq=freq, dl_name=dl_name, sampler=sampler,
                                    mean_final=mean_final, var_final=var_final)
        num_steps = x_tot.shape[0]
        if freq is None:
            freq = num_steps // min(num_steps, 50)

        if self.plot_level >= 1:
            x_tot_grid = x_tot[:, :self.num_plots_grid]
            name = str(i) + '_' + fb + '_' + str(n)
            im_dir = os.path.join(self.im_dir, name, self.prefix_fn(dl_name, sampler))
            gif_dir = os.path.join(self.gif_dir, self.prefix_fn(dl_name, sampler))

            os.makedirs(im_dir, exist_ok=True)
            os.makedirs(gif_dir, exist_ok=True)

            filename_grid = 'im_grid_start'
            filepath_grid_list = self.save_image(x_start[:self.num_plots_grid], filename_grid, im_dir, domain=0 if fb=='f' else 1)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            filename_grid = 'im_grid_last'
            filepath_grid_list = self.save_image(x_tot_grid[-1], filename_grid, im_dir, domain=1 if fb=='f' else 0)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            filename_grid = 'im_grid_data_x'
            filepath_grid_list = self.save_image(x_init[:self.num_plots_grid], filename_grid, im_dir, domain=0)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler) + filename_grid, filepath_grid_list, step=self.step, fb=fb)

            # Save y differently (no processing needed)
            filename_grid = 'im_grid_data_y'
            filepath_grid = os.path.join(im_dir, f'{filename_grid}.png')
            save_image(y_start[:self.num_plots_grid], filepath_grid, normalize=True, nrow=4)
            self.ipf.save_logger.log_image(self.prefix_fn(dl_name, sampler) + filename_grid, [filepath_grid], step=self.step, fb=fb)

            if self.plot_level >= 2:
                plot_paths = []
                x_start_tot_grid = torch.cat([x_start[:self.num_plots_grid].unsqueeze(0), x_tot_grid], dim=0)
                for k in range(num_steps+1):
                    if k % freq == 0 or k == num_steps:
                        # save png
                        filename_grid = 'im_grid_{0}'.format(k)
                        filepath_grid_list = self.save_image(x_start_tot_grid[k], filename_grid, im_dir, domain=1 if fb=='f' else 0)
                        plot_paths.append(filepath_grid_list)

                for d in [0, 1]:
                    make_gif([plot_path[d] for plot_path in plot_paths], output_directory=gif_dir, gif_name=f'{name}_dim_{d}_im_grid')

    def plot_and_record_batch_joint(self, x_start, y_start, x_last, x_init, x_last_true, iters, i, n, fb, dl_name='train', sampler='sde'):
        if fb == 'b' or self.ipf.transfer:            
            if self.plot_level >= 3:
                name = str(i) + '_' + fb + '_' + str(n)
                im_dir = os.path.join(self.im_dir, name, dl_name)
                inner_im_dir = os.path.join(im_dir, "im/")
                os.makedirs(inner_im_dir, exist_ok=True)
                
                file_idx = iters * self.ipf.accelerator.num_processes + self.ipf.accelerator.process_index

                filename = 'im_start'
                filepath = os.path.join(inner_im_dir, f'{filename}_{file_idx}.npy')
                np.save(filepath, x_start.cpu().numpy())

                filename = 'im_last'
                filepath = os.path.join(inner_im_dir, f'{filename}_{file_idx}.npy')
                np.save(filepath, x_last.cpu().numpy())

                filename = 'im_data_x'
                filepath = os.path.join(inner_im_dir, f'{filename}_{file_idx}.npy')
                np.save(filepath, x_init.cpu().numpy())

                filename = 'im_data_y'
                filepath = os.path.join(inner_im_dir, f'{filename}_{file_idx}.npy')
                np.save(filepath, y_start.cpu().numpy())

    def save_image(self, tensor, name, dir, domain=0):
        assert domain in [0, 1]
        fp_list = []
        if domain == 0:
            inverted_tensor, _ = self.ipf.init_ds.invert_preprocessing(tensor)
        else:
            inverted_tensor, _ = self.ipf.final_ds.invert_preprocessing(tensor)
        inverted_tensor = vutils.make_grid(inverted_tensor[:self.num_plots_grid], nrow=4)

        d = 0
        fp = os.path.join(dir, f'dim_{d}_{name}.png')
        plt.imsave(fp, inverted_tensor[0], vmin=-30, vmax=5, cmap='Blues_r')
        fp_list.append(fp)

        d = 1
        fp = os.path.join(dir, f'dim_{d}_{name}.png')
        plt.imsave(fp, inverted_tensor[1], vmin=-25, vmax=25, cmap='bwr_r')
        fp_list.append(fp)

        return fp_list