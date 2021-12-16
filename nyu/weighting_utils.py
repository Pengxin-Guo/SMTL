import torch, random
import numpy as np
import torch.nn.functional as F
from min_norm_solvers import MinNormSolver, gradient_normalizers

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def weight_update(weighting, loss_train, model, optimizer, epoch, batch_index, task_num,
                  clip_grad=False, scheduler=None, mgda_gn='none', 
                  random_distribution=None, avg_cost=None, mean=None, std=None, init_loss=None):
    """
    weighting: weight method (EW, GradNorm, UW, MGDA, DWA, GLS, PCGrad, GradDrop, IMTL, GradVac, random)
    mgda_gn: using in MGDA (none, l2, loss, loss+)
    random_distribution: using in random (uniform, normal, random_normal, inter_random, dirichlet, dropout, dropout_k)
    avg_cost: using in DWA
    mean, std: using in random_normal
    """
    batch_weight = None
    optimizer.zero_grad()
    if (weighting == 'PCGrad') or (weighting == 'GradVac'):
        optimizer.pc_backward(loss_train)
    elif weighting == 'MGDA_approx':
        batch_weight = model.MGDA_approx_backward(loss_train, mgda_gn=mgda_gn)
    elif weighting == 'GradNorm':
        model.GradNorm_backward(loss_train, init_loss)
        if (batch_index+1) % 50 == 0:
            print('{} weight: {}'.format(weighting, model.loss_scale))
    elif weighting == 'IMTL':
        batch_weight = model.IMTL_backward(loss_train)
    elif weighting == 'GradDrop':
        model.GradDrop_backward(loss_train)
    elif weighting == 'UW':
        loss = sum(1/(2*torch.exp(model.loss_scale[i]))*loss_train[i]+model.loss_scale[i]/2 for i in range(task_num))
        loss.backward()
        if (batch_index+1) % 200 == 0:
            print('{} weight: {}'.format(weighting, model.loss_scale))
    elif weighting == 'GLS':
        loss = torch.pow(loss_train.prod(), 1./task_num)
        loss.backward()
    elif weighting == 'WGLS':
        prod_loss = 1
        for t in range(task_num):
            prod_loss *= torch.pow(loss_train[t], model.loss_scale[t])
        loss = torch.pow(prod_loss, 1./model.loss_scale.sum())
        loss.backward()
#         print(model.loss_scale.grad)
        if (batch_index+1) % 20 == 0:
            print('{} weight: {}'.format(weighting, model.loss_scale))
    elif weighting == 'GLS_1':
        loss = torch.pow(loss_train.prod(), 1./task_num)/loss_train.sum()
        loss.backward()
    elif weighting == 'ULS':
        # no harmonic mean
        loss = task_num*loss_train.prod()/loss_train.sum()
        loss.backward()
    elif weighting == 'HLS':
        # harmonic mean
        loss = task_num/((1.0/loss_train).sum())
        loss.backward()
    else:
        if weighting == 'EW':
            batch_weight = torch.ones(task_num).cuda()
        elif weighting == 'MGDA':
            grads = {}
            loss_data = {}
            for i in range(task_num):
                grads[i] = list(torch.autograd.grad(loss_train[i], model.get_share_params(), retain_graph=True))
                loss_data[i] = loss_train[i].item()
            gn = gradient_normalizers(grads, loss_data, normalization_type=mgda_gn) # l2, loss, loss+, none
            for i in range(task_num):
                for g_i in range(len(grads[i])):
                    grads[i][g_i] = grads[i][g_i] / gn[i]
            sol, _ = MinNormSolver.find_min_norm_element([grads[i] for i in range(task_num)])
            batch_weight = torch.Tensor(sol).cuda()
        elif weighting == 'DWA' and avg_cost is not None:
            T = 2
            if epoch > 1:
                w_i = torch.Tensor(avg_cost[epoch-1]/avg_cost[epoch-2]).cuda()
                batch_weight = 3*F.softmax(w_i/T, dim=-1)
            else:
                batch_weight = torch.ones(task_num).cuda()
        elif weighting == 'random' and random_distribution is not None:
            if random_distribution == 'uniform':
                batch_weight = F.softmax(torch.rand(task_num).cuda(), dim=-1)
            elif random_distribution == 'normal':
                batch_weight = F.softmax(torch.randn(task_num).cuda(), dim=-1)
            elif random_distribution == 'inter_random':
                if random.randint(0, 1):
                    batch_weight = F.softmax(torch.rand(task_num).cuda(), dim=-1)
                else:
                    batch_weight = F.softmax(torch.randn(task_num).cuda(), dim=-1)
            elif random_distribution == 'dirichlet':
                # https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_number_generation
                alpha = 1
                gamma_sample = [random.gammavariate(alpha, 1) for _ in range(task_num)]
                dirichlet_sample = [v / sum(gamma_sample) for v in gamma_sample]
                batch_weight = torch.Tensor(dirichlet_sample).cuda()
            elif random_distribution == 'random_normal' and mean is not None and std is not None:
                batch_weight = F.softmax(torch.normal(mean, std).cuda(), dim=-1)
            elif random_distribution == 'GMM':
                mix = torch.distributions.Categorical(torch.ones(model.mix_k,))
                comp = torch.distributions.Independent(torch.distributions.Normal(model.comp_mu, model.comp_sigma), 1)
                gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)
                batch_weight = F.softmax(gmm.sample(), dim=-1).cuda()
            elif random_distribution == 'dropout':
                while True:
                    w = torch.randint(0, 2, (task_num,))
                    if w.sum()!=0:
                        batch_weight = w.cuda()
                        break
            elif len(random_distribution.split('_'))==2 and random_distribution.split('_')[0]=='dropout':
                w = random.sample(range(task_num), k=int(random_distribution.split('_')[1]))
                batch_weight = torch.zeros(task_num).cuda()
                batch_weight[w] = 1.
#                 while True:
#                     w = torch.randint(0, 2, (task_num,))
#                     if w.sum() == int(random_distribution.split('_')[1]):
#                         batch_weight = w.cuda()
#                         break
            else:
                raise('no support {}'.format(random_distribution))
        loss = torch.sum(loss_train*batch_weight)
#         optimizer.zero_grad()
        loss.backward()
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None and weighting != 'PCGrad' and weighting != 'GradVac':
        scheduler.step()
    if weighting != 'EW' and batch_weight is not None and (batch_index+1) % 20 == 0:
        print('{} weight: {}'.format(weighting, batch_weight.cpu().numpy()))
    return batch_weight