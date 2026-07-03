import os
import csv
import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint

from utils import misc


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))

    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch(epoch)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg['save_all']:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')

                save_checkpoint(
                    get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                    ckpt_name)

                if self.tester is not None:
                    self.logger.info("Test Epoch {}".format(self.epoch))
                    torch.cuda.empty_cache()
                    self.tester.inference()
                    cur_result = self.tester.evaluate()
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(
                            get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch),
                            ckpt_name)
                    self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

            progress_bar.update()

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        print(">>>>>>> Epoch:", str(epoch) + ":")

        use_consistency = self.cfg.get('use_consistency_loss', False)
        consistency_coef = self.cfg.get('consistency_loss_coef', 0.1)

        # CSV logging setup (consistency mode only)
        # csv_path = os.path.join(self.output_dir, 'consistency_log.csv')
        csv_path = "/nas2/data/heewon.hong/MonoIA_consistency/debug/consistency_log.csv"
        csv_header = ['epoch', 'iter', 'loss_cons', 'loss_det_A', 'loss_det_B',
                      'num_pairs', 'dim_diff', 'angle_diff']
        print(f"[CSV DEBUG] use_consistency={use_consistency}, csv_path={csv_path}")
        if use_consistency:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[CSV DEBUG] Creating/checking CSV file...")
            with open(csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                if os.path.getsize(csv_path) == 0:
                    w.writerow(csv_header)
            print(f"[CSV DEBUG] CSV ready at: {os.path.abspath(csv_path)}")

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, batch in enumerate(self.train_loader):

            if use_consistency:
                # ============ consistency training path ============
                (inputs_A, calibs_A, targets_A_raw, info_A), \
                (inputs_B, calibs_B, targets_B_raw, info_B) = batch

                inputs_A = inputs_A.to(self.device)
                calibs_A = calibs_A.to(self.device)
                inputs_B = inputs_B.to(self.device)
                calibs_B = calibs_B.to(self.device)
                for key in targets_A_raw.keys():
                    targets_A_raw[key] = targets_A_raw[key].to(self.device)
                for key in targets_B_raw.keys():
                    targets_B_raw[key] = targets_B_raw[key].to(self.device)

                img_sizes_A = targets_A_raw['img_size']
                img_sizes_B = targets_B_raw['img_size']
                # save mask_2d before prepare_targets to derive valid_obj_raw_idx later
                mask_2d_A = targets_A_raw['mask_2d']
                mask_2d_B = targets_B_raw['mask_2d']

                targets_A_list = self.prepare_targets(targets_A_raw, inputs_A.shape[0])
                targets_B_list = self.prepare_targets(targets_B_raw, inputs_B.shape[0])

                dn_args_A, dn_args_B = None, None
                if self.cfg["use_dn"]:
                    dn_args_A = (targets_A_list, self.cfg['scalar'], self.cfg['label_noise_scale'],
                                 self.cfg['box_noise_scale'], self.cfg['num_patterns'])
                    dn_args_B = (targets_B_list, self.cfg['scalar'], self.cfg['label_noise_scale'],
                                 self.cfg['box_noise_scale'], self.cfg['num_patterns'])

                self.optimizer.zero_grad()

                outputs_A = self.model(inputs_A, calibs_A, targets_A_raw, img_sizes_A, dn_args=dn_args_A)
                outputs_B = self.model(inputs_B, calibs_B, targets_B_raw, img_sizes_B, dn_args=dn_args_B)

                losses_dict_A = self.detr_loss(outputs_A, targets_A_list)
                losses_dict_B = self.detr_loss(outputs_B, targets_B_list)

                loss_cons, cons_stats = self.compute_consistency_loss(
                    outputs_A, outputs_B,
                    targets_A_list, targets_B_list,
                    mask_2d_A, mask_2d_B,
                )

                weight_dict = self.detr_loss.weight_dict
                loss_det_A = sum(losses_dict_A[k] * weight_dict[k] for k in losses_dict_A if k in weight_dict)
                loss_det_B = sum(losses_dict_B[k] * weight_dict[k] for k in losses_dict_B if k in weight_dict)
                total_loss = loss_det_A + loss_det_B + consistency_coef * loss_cons

                # CSV logging every 8 iterations
                if batch_idx % 8 == 0:
                    with open(csv_path, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            epoch,
                            batch_idx,
                            round(loss_cons.item(), 6),
                            round(loss_det_A.item(), 4),
                            round(loss_det_B.item(), 4),
                            cons_stats['num_pairs'],
                            round(cons_stats['dim_diff'], 6),
                            round(cons_stats['angle_diff'], 6),
                        ])

                # console logging
                if batch_idx % 30 == 0:
                    print("----", batch_idx, "----")
                    print("loss_A: %.2f, loss_B: %.2f, loss_cons: %.4f, num_pairs: %d" % (
                        loss_det_A.item(), loss_det_B.item(), loss_cons.item(), cons_stats['num_pairs']))

                total_loss.backward()
                self.optimizer.step()

            else:
                # ============ original single-focal path (unchanged) ============
                inputs, calibs, targets, info = batch
                inputs = inputs.to(self.device)
                calibs = calibs.to(self.device)
                for key in targets.keys():
                    targets[key] = targets[key].to(self.device)
                img_sizes = targets['img_size']
                original_targets = targets
                targets = self.prepare_targets(targets, inputs.shape[0])

                dn_args = None
                if self.cfg["use_dn"]:
                    dn_args = (targets, self.cfg['scalar'], self.cfg['label_noise_scale'],
                               self.cfg['box_noise_scale'], self.cfg['num_patterns'])

                self.optimizer.zero_grad()
                outputs = self.model(inputs, calibs, original_targets, img_sizes, dn_args=dn_args)
                mask_dict = None
                detr_losses_dict = self.detr_loss(outputs, targets, mask_dict)

                weight_dict = self.detr_loss.weight_dict
                detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict]
                detr_losses = sum(detr_losses_dict_weighted)

                detr_losses_dict = misc.reduce_dict(detr_losses_dict)
                detr_losses_dict_log = {}
                detr_losses_log = 0
                for k in detr_losses_dict.keys():
                    if k in weight_dict:
                        detr_losses_dict_log[k] = (detr_losses_dict[k] * weight_dict[k]).item()
                        detr_losses_log += detr_losses_dict_log[k]
                detr_losses_dict_log["loss_detr"] = detr_losses_log

                flags = [True] * 5
                if batch_idx % 30 == 0:
                    print("----", batch_idx, "----")
                    print("%s: %.2f, " % ("loss_detr", detr_losses_dict_log["loss_detr"]))
                    for key, val in detr_losses_dict_log.items():
                        if key == "loss_detr":
                            continue
                        if "0" in key or "1" in key or "2" in key or "3" in key or "4" in key or "5" in key:
                            if flags[int(key[-1])]:
                                print("")
                                flags[int(key[-1])] = False
                        print("%s: %.2f, " % (key, val), end="")
                    print("")
                    print("")

                detr_losses.backward()
                self.optimizer.step()

            progress_bar.update()
        progress_bar.close()

    def compute_consistency_loss(self, outputs_A, outputs_B, targets_A_list, targets_B_list, mask_2d_A, mask_2d_B):
        """
        For each batch item, finds objects visible in both focal versions (via mask_2d),
        then compares geometry-invariant predictions (pred_3d_dim, pred_angle) for the
        same physical objects matched by the Hungarian matcher (group_num=1).
        """
        batch_size = mask_2d_A.shape[0]

        # use group_num=1 for clean one-to-one query-to-GT assignments
        indices_A = self.detr_loss.matcher(outputs_A, targets_A_list, group_num=1)
        indices_B = self.detr_loss.matcher(outputs_B, targets_B_list, group_num=1)

        total_loss = outputs_A['pred_3d_dim'].new_zeros(1).squeeze()
        num_pairs = 0
        dim_diff_sum = 0.0
        angle_diff_sum = 0.0

        for b in range(batch_size):
            # raw array indices of valid objects in each focal version
            raw_A = mask_2d_A[b].nonzero(as_tuple=False).view(-1)  # (num_valid_A,)
            raw_B = mask_2d_B[b].nonzero(as_tuple=False).view(-1)  # (num_valid_B,)

            if len(raw_A) == 0 or len(raw_B) == 0:
                continue

            raw_A_np = raw_A.cpu().numpy()
            raw_B_np = raw_B.cpu().numpy()

            # physical objects visible in both focal versions
            common_raw = np.intersect1d(raw_A_np, raw_B_np)
            if len(common_raw) == 0:
                continue

            src_A, tgt_A = indices_A[b]  # (matched_query_idx, matched_gt_pos)
            src_B, tgt_B = indices_B[b]

            tgt_A_np = tgt_A.cpu().numpy()
            tgt_B_np = tgt_B.cpu().numpy()

            for raw_idx in common_raw:
                # position in filtered targets list (after prepare_targets mask filtering)
                pos_in_A = int((raw_A_np == raw_idx).nonzero()[0][0])
                pos_in_B = int((raw_B_np == raw_idx).nonzero()[0][0])

                # find which query the matcher assigned to this GT position
                match_A = (tgt_A_np == pos_in_A).nonzero()[0]
                match_B = (tgt_B_np == pos_in_B).nonzero()[0]

                if len(match_A) == 0 or len(match_B) == 0:
                    continue

                query_A = src_A[match_A[0]].item()
                query_B = src_B[match_B[0]].item()

                # 3D size consistency
                dim_A = outputs_A['pred_3d_dim'][b, query_A]
                dim_B = outputs_B['pred_3d_dim'][b, query_B]
                dim_diff = F.l1_loss(dim_A, dim_B)
                total_loss = total_loss + dim_diff
                dim_diff_sum += dim_diff.item()

                # orientation consistency
                angle_A = outputs_A['pred_angle'][b, query_A]
                angle_B = outputs_B['pred_angle'][b, query_B]
                angle_diff = F.l1_loss(angle_A, angle_B)
                total_loss = total_loss + angle_diff
                angle_diff_sum += angle_diff.item()

                num_pairs += 1

        zero = outputs_A['pred_3d_dim'].new_zeros(1).squeeze()
        if num_pairs == 0:
            return zero, {'num_pairs': 0, 'dim_diff': 0.0, 'angle_diff': 0.0}

        stats = {
            'num_pairs': num_pairs,
            'dim_diff': dim_diff_sum / num_pairs,
            'angle_diff': angle_diff_sum / num_pairs,
        }
        return total_loss / num_pairs, stats

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
                if key == "depth_map" or key == "target_focals" or key == "obj_region" or key == "target_focals":
                    target_dict[key] = val[bz]
            targets_list.append(target_dict)
        return targets_list
