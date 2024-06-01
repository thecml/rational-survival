import torch
import rational.encoder as encoder
import rational.generator as generator 
import rational.tagger as tagger
import rational.empty as empty 
import rational.learn as learn
import os
import pdb

def get_model(args, n_features):
    if args.snapshot is None:
        if args.use_as_tagger == True:
            gen = empty.Empty()
            model = tagger.TaggerTab(args)
        else:
            gen = generator.Generator(n_features, args)
            model = encoder.Encoder(n_features, args)
    else:
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            gen_path = learn.get_gen_path(args.snapshot)
            if os.path.exists(gen_path):
                gen   = torch.load(gen_path)
            model = torch.load(args.snapshot)
        except :
            print("Sorry, This snapshot doesn't exist."); exit()
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=range(args.num_gpus))
        if not gen is None:
            gen = torch.nn.DataParallel(gen, device_ids=range(args.num_gpus))
    return gen, model
