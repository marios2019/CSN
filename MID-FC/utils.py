import numpy as np 
import os, sys
import pandas as pd 
import torch
names     = ['Bed', 'Bottle', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 
             'Earphone', 'Faucet', 'Knife', 'Lamp', 'Microwave', 'Refrigerator',
             'StorageFurniture', 'Table', 'TrashCan', 'Vase']

def create_csv(logs_dir):
    logs_dir = 'logs/backbone_ssa_fc_logit_n_heads_1_cosdecay'

    columns = []
    data = []
    for name in names:
        path = os.path.join(logs_dir, name, 'test_summaries.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        columns.append(name)
        iou = df[name].tolist()[-1]
        data.append(iou)

    avg_iou = sum(data) / len(data)
    print(avg_iou)

    summ = pd.DataFrame([data], columns=columns)
    print(summ)

def load_trained_ssa_layers(model, logs_dir):
    saved_logs_dir = logs_dir ## ssa weights.
    path = os.path.join(saved_logs_dir, 'trained_layers.pth')
    ckpt = torch.load(path)
    for k, v in ckpt.items():
        model.state_dict()[k].copy_(torch.clone(v))
    del ckpt
    ckpt = None
    torch.cuda.empty_cache()
    print('trained model imported from', path)
    return model 

names     = ['Bed', 'Bottle', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 
             'Earphone', 'Faucet', 'Knife', 'Lamp', 'Microwave', 'Refrigerator',
             'StorageFurniture', 'Table', 'TrashCan', 'Vase']

def accumulate_predictions(partname):   

    dataroot = f'/work/siddhantgarg_umass_edu/int-vis/O-CNN/tensorflow/script/logs/partnet_weights_and_logs/partnet_finetune/test_data_features/{partname}' # change to your data root
    dataroot_new = f'/work/siddhantgarg_umass_edu/int-vis/O-CNN/tensorflow/script/logs/partnet_weights_and_logs/partnet_finetune/test_data_for_paper/{partname}' # change to your data root

    render_dir = f'logs/rendering/{partname}'
    if not os.path.exists(render_dir):
        os.mkdir(render_dir)
        print(render_dir, 'created!')

    pts_path = os.path.join(dataroot_new, 'pts')
    labels_path = os.path.join(dataroot_new, 'point_labels')
    midfc_pred_path = os.path.join(dataroot_new, 'midfc_pred')
    midfc_ssa_path = os.path.join(dataroot_new, 'midfc_ssa')
    midfc_csa_path = os.path.join(dataroot_new, 'midfc_csa_K_4')

    n_shapes = len(os.listdir(pts_path))
    
    for i in range(n_shapes):
        print(f'{partname}: {i}/{n_shapes}')
        pts = np.load(os.path.join(pts_path, f'shape_{i}.npy'))
        gt = np.load(os.path.join(labels_path, f'shape_{i}.npy'))
        gt = np.expand_dims(gt, axis=1)
        midfc_pred = np.load(os.path.join(midfc_pred_path, f'shape_{i}.npy'))
        midfc_pred = np.expand_dims(midfc_pred, axis=1)
        ssa = np.load(os.path.join(midfc_ssa_path, f'shape_{i}.npy'))
        ssa = np.expand_dims(ssa, axis=1)
        csa = np.load(os.path.join(midfc_csa_path, f'shape_{i}.npy'))
        csa = np.expand_dims(csa, axis=1)

        a = np.concatenate((pts, gt, midfc_pred, ssa, csa), axis=1)

        path = os.path.join(render_dir, f'shape_{i}.npy')
        with open(path, 'wb') as f:
            np.save(f, a)

def accumulate_neigh(partname):
    render_dir = f'logs/rendering_neighbors/{partname}'
    if not os.path.exists(render_dir):
        os.mkdir(render_dir)
        print(render_dir, 'created!')

    dataroot = f'logs/train_data_for_paper/{partname}'
    pts_path = os.path.join(dataroot, 'pts')
    labels_path = os.path.join(dataroot, 'point_labels')

    shapes = os.listdir(pts_path)

    for sh in shapes:
        pts = np.load(os.path.join(pts_path, sh))
        labels = np.load(os.path.join(labels_path, sh))
        labels = np.expand_dims(labels, axis=1)

        a = np.concatenate((pts, labels), axis=1)

        path = os.path.join(render_dir, sh)
        print(a.shape, path)
        # sys.exit()
        with open(path, 'wb') as f:
            np.save(f, a)

# accumulate_neigh('Table')
# accumulate_neigh('Lamp')
# accumulate_neigh('Chair')
# accumulate_neigh('Table')
# model = load_trained_ssa_layers('logs/backbone_fc_csa_logit_n_heads_1/Bed', 'logs/backbone_fc_ssa_logit_n_heads_1_cosdecay/Bed')

# a = '61.77957403	51.91555023	58.80114039	55.7532231	57.45967428	76.14864111	91.52489305	58.53244066	56.13431136	64.91309404	57.31393695	32.32345879	79.3842574	64.43816423	62.88832426	45.76647679	66.60519441	70.34997741'
# a = '61.86285223	52.30797082	58.64417432	55.70840237	57.75076238	76.31111579	91.37571785	58.99980186	54.20140897	64.97185656	60.04070346	32.93472092	79.05371639	63.97509578	62.91155988	45.98731684	66.5699427	69.92422102'
# a = '61.86337981	52.12851403	58.68388306	55.70870709	57.36478687	76.34272543	91.37621031	58.91509693	54.48684723	65.13142327	60.07530882	32.36422008	78.95819722	64.06221555	62.89504688	45.82581816	67.45524727	69.90320859'
# a = '62.04520165	52.29206146	58.62682543	55.70957566	57.92261741	76.40665508	91.38854795	58.82855403	54.4039421	65.12612365	61.66992415	33.13671085	79.07560921	64.00310087	62.88566771	45.84951624	67.60036039	69.84263577'
# a = '62.05494503	52.23403501	58.61831587	55.7061782	57.65411079	76.36580264	91.39366701	58.87723631	54.49589946	65.18266424	62.16550237	33.08029687	79.17267176	64.0354726	62.88292645	45.97731023	67.21934805	69.87262764'
# a = '62.02385424	52.25360771	58.55251953	55.69457238	57.60289937	76.39937451	91.39167349	58.89809229	54.50551076	65.19937185	62.15945271	33.07621392	79.1163488	64.00075379	62.88374428	45.91966917	66.84688213	69.90483539'
# a = a.split('\t')
# a = [float(x) for x in a]
# # print(a)
# s = ''
# for x in a[1:]:
#     s += f'{x:.1f} & '
# s += f'{a[0]:.1f} & '
# print(s)