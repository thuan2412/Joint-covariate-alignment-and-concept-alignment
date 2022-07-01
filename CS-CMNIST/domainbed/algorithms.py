import copy

import numpy as np
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torch.nn.functional as F

import networks
# from ..domainbed.lib.misc import random_pairs_of_minibatches

ALGORITHMS = [
    #'ERM',
    #'IRM',
    #'IBERM',
    'IBIRM',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():  # global() is a dict
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

###### 1. ERM algorithm
class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.num_classes = num_classes

        self.xyl = False
        if hparams["xylopt"]:
            print("LRRRRRRR:", self.hparams["lr"])
            self.xyl = True
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.hparams["lr"],
                momentum=0.9, weight_decay=self.hparams['weight_decay'])

            print("DECAY STEP:", hparams["sch_size"])
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=hparams["sch_size"], gamma=0.1)
        else:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.xyl:
            self.scheduler.step()
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


###### 2. IRM algorithm
class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.normalize = hparams['normalize']

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.).to(y.device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches):
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                    >= self.hparams['irm_penalty_anneal_iters'] else 0.0)  # todo

        nll = 0.
        penalty = 0.
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        if self.normalize and penalty_weight > 1:
            loss /= (1 + penalty_weight)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            if not self.xyl:
                print("!!!!UPDATE IRM ADAM OPTIMIZER")
                self.optimizer = torch.optim.Adam(
                    self.network.parameters(),
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.xyl:
            self.scheduler.step()
            pass

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}

##### 3. IB_ERM algorithm
# class IBERM(ERM):
#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(IBERM, self).__init__(input_shape, num_classes, num_domains,
#                                    hparams)
#         self.register_buffer('update_count', torch.tensor([0]))
#         self.normalize = hparams['normalize']

#     def update(self, minibatches):
#         penalty_weight = (self.hparams['ib_lambda'] if self.update_count
#                        >= self.hparams['ib_penalty_anneal_iters'] else 0.0)
#         all_x = torch.cat([x for x, y in minibatches])
#         all_y = torch.cat([y for x, y in minibatches])

#         logits = self.featurizer(all_x)
#         loss = F.cross_entropy(self.classifier(logits), all_y)

#         var_loss = logits.var(dim=0).mean()

#         loss += penalty_weight * var_loss
#         if self.normalize and penalty_weight > 1:
#             loss /= (1 + penalty_weight)

#         if self.update_count == self.hparams['ib_penalty_anneal_iters']:
#             if not self.xyl:
#                 print("!!!!UPDATE IB-ERM ADAM OPTIMIZER")
#                 self.optimizer = torch.optim.Adam(
#                     self.network.parameters(),
#                     lr=self.hparams["lr"],
#                     weight_decay=self.hparams['weight_decay'])

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         if self.xyl:
#             self.scheduler.step()
#         return {'loss': loss.item(), "var": var_loss.item()}


##### 4. IB_IRM algorithm
# class IBIRM(IRM):
#     """Invariant Risk Minimization"""

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(IBIRM, self).__init__(input_shape, num_classes, num_domains,
#                                   hparams)

#     def update(self, minibatches):
#         penalty_weight = (self.hparams['irm_lambda'] if self.update_count
#                     >= self.hparams['irm_penalty_anneal_iters'] else 0.0)  # todo

#         ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
#                    >= self.hparams['ib_penalty_anneal_iters'] else 0.0)

#         nll = 0.
#         penalty = 0.
#         all_x = torch.cat([x for x, y in minibatches])
#         all_y = torch.cat([y for x, y in minibatches])
#         inter_logits = self.featurizer(all_x)
#         all_logits = self.classifier(inter_logits)
#         all_logits_idx = 0
#         for i, (x, y) in enumerate(minibatches):
#             logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
#             all_logits_idx += x.shape[0]
#             nll += F.cross_entropy(logits, y)
#             penalty += self._irm_penalty(logits, y)
#         nll /= len(minibatches)
#         penalty /= len(minibatches)
#         loss = nll + (penalty_weight * penalty)
#         if self.normalize and penalty_weight > 1:
#             loss /= (1 + penalty_weight)

#         var_loss = inter_logits.var(dim=0).mean()
#         loss += ib_penalty_weight * var_loss

#         if self.update_count == self.hparams['irm_penalty_anneal_iters']:
#             if not self.xyl:
#                 print("!!!!UPDATE IB-ERM ADAM OPTIMIZER")
#                 self.optimizer = torch.optim.Adam(
#                     self.network.parameters(),
#                     lr=self.hparams["lr"],
#                     weight_decay=self.hparams['weight_decay'])

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         if self.xyl:
#             self.scheduler.step()
#             pass

#         self.update_count += 1
#         return {'loss': loss.item(), 'nll': nll.item(),
#                 'penalty': penalty.item(),  "var": var_loss.item()}

###### 5. CEM-CORAL algorithm
# class IBIRM(IRM): 
#     """Invariant Risk Minimization"""

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#        super(IBIRM, self).__init__(input_shape, num_classes, num_domains,
#                                  hparams)
#     def my_cdist(self, x1, x2):
#         x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
#         x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
#         res = torch.addmm(x2_norm.transpose(-2, -1),
#                           x1,
#                           x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
#         return res.clamp_min_(1e-30)

#     def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
#                                            1000]):
#         D = self.my_cdist(x, y)
#         K = torch.zeros_like(D)

#         for g in gamma:
#             K.add_(torch.exp(D.mul(-g)))

#         return K

#     def mmd(self, x, y): ### this for CORAL - fix for CORAL
#         mean_x = x.mean(0, keepdim=True)
#         mean_y = y.mean(0, keepdim=True)
#         cent_x = x - mean_x
#         cent_y = y - mean_y
#         cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
#         cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

#         mean_diff = (mean_x - mean_y).pow(2).mean()
#         cova_diff = (cova_x - cova_y).pow(2).mean()

#         return mean_diff + cova_diff

#     def update(self, minibatches):
#         penalty_weight = (self.hparams['irm_lambda'] if self.update_count
#                     >= self.hparams['irm_penalty_anneal_iters'] else 0.0)  # todo

#         ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
#                    >= self.hparams['ib_penalty_anneal_iters'] else 0.0)

#         mmd_penalty_weight = (self.hparams['mmd_lambda'] if self.update_count
#                    >= self.hparams['mmd_penalty_anneal_iters'] else 0.0)

#         nll = 0.
#         penalty = 0.
#         all_x = torch.cat([x for x, y in minibatches])
#         all_y = torch.cat([y for x, y in minibatches])


#         inter_logits = self.featurizer(all_x)
#         inter_logits = F.normalize(inter_logits, dim=-1)

#         all_logits = self.classifier(inter_logits)
#         all_logits_idx = 0
#         for i, (x, y) in enumerate(minibatches):
#             logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
#             all_logits_idx += x.shape[0]
#             nll += F.cross_entropy(logits, y)
#             penalty += self._irm_penalty(logits, y)
#         nll /= len(minibatches)
#         penalty /= len(minibatches)
#         loss = nll + (penalty_weight * penalty)
#         if self.normalize and penalty_weight > 1:
#             loss /= (1 + penalty_weight)
        
#         class_loss = F.cross_entropy(logits,y).div(math.log(2)) ### H(Y|Z)    
#         entropy_loss = inter_logits.var(dim=0).mean() ### H(Z)
#         var_loss = entropy_loss + class_loss
#         loss += ib_penalty_weight * var_loss

#         #### MMD loss
#         mmd_loss = 0
#         nmb = len(minibatches)

#         features = [self.featurizer(xi) for xi, _ in minibatches]
       
#         for i in range(nmb):
#             for j in range(i + 1, nmb):
#                 mmd_loss += self.mmd(features[i], features[j])

#         if nmb > 1:
#             mmd_loss /= (nmb * (nmb - 1) / 2)

#         ## final loss
#         loss += mmd_penalty_weight*mmd_loss/100 


#         if self.update_count == self.hparams['irm_penalty_anneal_iters']:
#             if not self.xyl:
#                 print("!!!!UPDATE IB-ERM ADAM OPTIMIZER")
#                 self.optimizer = torch.optim.Adam(
#                     self.network.parameters(),
#                     lr=self.hparams["lr"],
#                     weight_decay=self.hparams['weight_decay'])

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         if self.xyl:
#             self.scheduler.step()
#             pass

#         self.update_count += 1
#         return {'loss': loss.item(), 'nll': nll.item(),
#                 'penalty': penalty.item(),  "var": var_loss.item()}

###### 6. CEM-MMD algorithm
class IBIRM(IRM): 
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
       super(IBIRM, self).__init__(input_shape, num_classes, num_domains,
                                 hparams)
    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y): ### using Gaussian kernel here - fix for MMD 
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def update(self, minibatches):
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                    >= self.hparams['irm_penalty_anneal_iters'] else 0.0)  # todo

        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                   >= self.hparams['ib_penalty_anneal_iters'] else 0.0)

        mmd_penalty_weight = (self.hparams['mmd_lambda'] if self.update_count
                   >= self.hparams['mmd_penalty_anneal_iters'] else 0.0)

        nll = 0.
        penalty = 0.
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])


        inter_logits = self.featurizer(all_x)
        inter_logits = F.normalize(inter_logits, dim=-1)

        all_logits = self.classifier(inter_logits)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        if self.normalize and penalty_weight > 1:
            loss /= (1 + penalty_weight)
        
        class_loss = F.cross_entropy(logits,y).div(math.log(2)) ### H(Y|Z)    
        entropy_loss = inter_logits.var(dim=0).mean() ### H(Z)
        var_loss = entropy_loss + class_loss
        loss += ib_penalty_weight * var_loss

        #### MMD loss
        mmd_loss = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
       
        for i in range(nmb):
            for j in range(i + 1, nmb):
                mmd_loss += self.mmd(features[i], features[j])

        if nmb > 1:
            mmd_loss /= (nmb * (nmb - 1) / 2)

        ## final loss
        loss += mmd_penalty_weight*mmd_loss/100 


        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            if not self.xyl:
                print("!!!!UPDATE IB-ERM ADAM OPTIMIZER")
                self.optimizer = torch.optim.Adam(
                    self.network.parameters(),
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.xyl:
            self.scheduler.step()
            pass

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item(),  "var": var_loss.item()}

###### 7. IRM-MMD algorithm
# class IBIRM(IRM): 
#     """Invariant Risk Minimization"""

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#        super(IBIRM, self).__init__(input_shape, num_classes, num_domains,
#                                  hparams)
#     def my_cdist(self, x1, x2):
#         x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
#         x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
#         res = torch.addmm(x2_norm.transpose(-2, -1),
#                           x1,
#                           x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
#         return res.clamp_min_(1e-30)

#     def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
#                                            1000]):
#         D = self.my_cdist(x, y)
#         K = torch.zeros_like(D)

#         for g in gamma:
#             K.add_(torch.exp(D.mul(-g)))

#         return K

#     def mmd(self, x, y): ### using Gaussian kernel here - fix for MMD 
#         Kxx = self.gaussian_kernel(x, x).mean()
#         Kyy = self.gaussian_kernel(y, y).mean()
#         Kxy = self.gaussian_kernel(x, y).mean()
#         return Kxx + Kyy - 2 * Kxy


#     def update(self, minibatches):
#         penalty_weight = (self.hparams['irm_lambda'] if self.update_count
#                     >= self.hparams['irm_penalty_anneal_iters'] else 0.0)  # todo

#         ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
#                    >= self.hparams['ib_penalty_anneal_iters'] else 0.0)

#         mmd_penalty_weight = (self.hparams['mmd_lambda'] if self.update_count
#                    >= self.hparams['mmd_penalty_anneal_iters'] else 0.0)

#         nll = 0.
#         penalty = 0.
#         all_x = torch.cat([x for x, y in minibatches])
#         all_y = torch.cat([y for x, y in minibatches])


#         inter_logits = self.featurizer(all_x)
#         inter_logits = F.normalize(inter_logits, dim=-1)

#         all_logits = self.classifier(inter_logits)
#         all_logits_idx = 0
#         for i, (x, y) in enumerate(minibatches):
#             logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
#             all_logits_idx += x.shape[0]
#             nll += F.cross_entropy(logits, y)
#             penalty += self._irm_penalty(logits, y)
#         nll /= len(minibatches)
#         penalty /= len(minibatches)
#         loss = nll + (penalty_weight * penalty)
#         if self.normalize and penalty_weight > 1:
#             loss /= (1 + penalty_weight)
        
#         class_loss = F.cross_entropy(logits,y).div(math.log(2)) ### H(Y|Z)    
#         entropy_loss = inter_logits.var(dim=0).mean() ### H(Z)
#         var_loss = entropy_loss + class_loss
#         loss += ib_penalty_weight * var_loss * 0 #### since only want IRM-MMD, ib_penalty (conditional entropy) loss must be 0.

#         #### MMD loss
#         mmd_loss = 0
#         nmb = len(minibatches)

#         features = [self.featurizer(xi) for xi, _ in minibatches]
       
#         for i in range(nmb):
#             for j in range(i + 1, nmb):
#                 mmd_loss += self.mmd(features[i], features[j])

#         if nmb > 1:
#             mmd_loss /= (nmb * (nmb - 1) / 2)

#         ## final loss
#         loss += mmd_penalty_weight*mmd_loss/100 


#         if self.update_count == self.hparams['irm_penalty_anneal_iters']:
#             if not self.xyl:
#                 print("!!!!UPDATE IB-ERM ADAM OPTIMIZER")
#                 self.optimizer = torch.optim.Adam(
#                     self.network.parameters(),
#                     lr=self.hparams["lr"],
#                     weight_decay=self.hparams['weight_decay'])

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         if self.xyl:
#             self.scheduler.step()
#             pass

#         self.update_count += 1
#         return {'loss': loss.item(), 'nll': nll.item(),
#                 'penalty': penalty.item(),  "var": var_loss.item()}

# ###### 8. CEM algorithm
# class IBIRM(IRM): 
#     """Invariant Risk Minimization"""

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#        super(IBIRM, self).__init__(input_shape, num_classes, num_domains,
#                                  hparams)
#     def my_cdist(self, x1, x2):
#         x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
#         x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
#         res = torch.addmm(x2_norm.transpose(-2, -1),
#                           x1,
#                           x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
#         return res.clamp_min_(1e-30)

#     def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
#                                            1000]):
#         D = self.my_cdist(x, y)
#         K = torch.zeros_like(D)

#         for g in gamma:
#             K.add_(torch.exp(D.mul(-g)))

#         return K

#     def mmd(self, x, y): ### using Gaussian kernel here - fix for MMD 
#         Kxx = self.gaussian_kernel(x, x).mean()
#         Kyy = self.gaussian_kernel(y, y).mean()
#         Kxy = self.gaussian_kernel(x, y).mean()
#         return Kxx + Kyy - 2 * Kxy

#     def update(self, minibatches):
#         penalty_weight = (self.hparams['irm_lambda'] if self.update_count
#                     >= self.hparams['irm_penalty_anneal_iters'] else 0.0)  # todo

#         ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
#                    >= self.hparams['ib_penalty_anneal_iters'] else 0.0)

#         mmd_penalty_weight = (self.hparams['mmd_lambda'] if self.update_count
#                    >= self.hparams['mmd_penalty_anneal_iters'] else 0.0)

#         nll = 0.
#         penalty = 0.
#         all_x = torch.cat([x for x, y in minibatches])
#         all_y = torch.cat([y for x, y in minibatches])


#         inter_logits = self.featurizer(all_x)
#         inter_logits = F.normalize(inter_logits, dim=-1)

#         all_logits = self.classifier(inter_logits)
#         all_logits_idx = 0
#         for i, (x, y) in enumerate(minibatches):
#             logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
#             all_logits_idx += x.shape[0]
#             nll += F.cross_entropy(logits, y)
#             penalty += self._irm_penalty(logits, y)
#         nll /= len(minibatches)
#         penalty /= len(minibatches)
#         loss = nll + (penalty_weight * penalty)
#         if self.normalize and penalty_weight > 1:
#             loss /= (1 + penalty_weight)
        
#         class_loss = F.cross_entropy(logits,y).div(math.log(2)) ### H(Y|Z)    
#         entropy_loss = inter_logits.var(dim=0).mean() ### H(Z)
#         var_loss = entropy_loss + class_loss
#         loss += ib_penalty_weight * var_loss

#         #### MMD loss
#         mmd_loss = 0
#         nmb = len(minibatches)

#         features = [self.featurizer(xi) for xi, _ in minibatches]
       
#         for i in range(nmb):
#             for j in range(i + 1, nmb):
#                 mmd_loss += self.mmd(features[i], features[j])

#         if nmb > 1:
#             mmd_loss /= (nmb * (nmb - 1) / 2)

#         ## final loss
#         loss += mmd_penalty_weight*mmd_loss*0 ### since we do not need MMD loss for CEM, just IRM loss and conditional entropy loss


#         if self.update_count == self.hparams['irm_penalty_anneal_iters']:
#             if not self.xyl:
#                 print("!!!!UPDATE IB-ERM ADAM OPTIMIZER")
#                 self.optimizer = torch.optim.Adam(
#                     self.network.parameters(),
#                     lr=self.hparams["lr"],
#                     weight_decay=self.hparams['weight_decay'])

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         if self.xyl:
#             self.scheduler.step()
#             pass

#         self.update_count += 1
#         return {'loss': loss.item(), 'nll': nll.item(),
#                 'penalty': penalty.item(),  "var": var_loss.item()}