import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import pdb

def get_train_loader(train_data, args):
    if args.class_balance:
        sampler = data.sampler.WeightedRandomSampler(
                weights=train_data.weights,
                num_samples=len(train_data),
                replacement=True)
        train_loader = data.DataLoader(
                train_data,
                num_workers=args.num_workers,
                sampler=sampler,
                batch_size=args.batch_size)
    else:
        train_loader = data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False)

    return train_loader

def get_rationales_text(mask, text):
    if mask is None:
        return text
    masked_text = []
    for i, t in enumerate(text):
        sample_mask = list(mask.data[i])
        original_words = t.split()
        words = [ w if m  > .5 else "_" for w,m in zip(original_words, sample_mask) ]
        masked_sample = " ".join(words)
        masked_text.append(masked_sample)
    return masked_text

def get_rationales(mask):
    ft_names = list()
    for sample in mask:
        ft_indicies = np.argwhere(sample > 0.5)
        ft_names.append([f'x_{idx+1}' for idx in ft_indicies.squeeze(0).numpy()])
    return ft_names
    
def get_valid_loader(valid_data, args):
    valid_loader = data.DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)
    return valid_loader

def get_optimizer(models, args):
    '''
        -models: List of models (such as Generator, classif, memory, etc)
        -args: experiment level config

        returns: torch optimizer over models
    '''
    params = []
    for model in models:
        params.extend([param for param in model.parameters() if param.requires_grad])
    return torch.optim.Adam(params, lr=args.lr,  weight_decay=args.weight_decay)


def get_x_indx(batch, args, eval_model):
    x_indx = autograd.Variable(batch['x'], volatile=eval_model)
    return x_indx



def get_hard_mask(z, return_ind=False):
    '''
        -z: torch Tensor where each element probablity of element
        being selected
        -args: experiment level config

        returns: A torch variable that is binary mask of z >= .5
    '''
    max_z, ind = torch.max(z, dim=-1)
    if return_ind:
        del z
        return ind
    masked = torch.ge(z, 0.5).float()
    del z
    return masked

def get_gen_path(model_path):
    '''
        -model_path: path of encoder model

        returns: path of generator
    '''
    return '{}.gen'.format(model_path)

def one_hot(label, num_class):
    vec = torch.zeros( (1, num_class) )
    vec[0][label] = 1
    return vec


def gumbel_softmax(input, temperature, cuda):
    noise = torch.rand(input.size())
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    noise = autograd.Variable(noise)
    if cuda:
        noise = noise.cuda()
    x = (input + noise) / temperature
    x = F.softmax(x.view(-1,  x.size()[-1]), dim=-1)
    return x.view_as(input)
