_base_ = [
    '../_base_/models/fccdn_se.py', 
    '../common/fccdn_512x512_40k_my-seg-data.py']
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = None

model = dict(type='SDIEncoderDecoder', 
            pretrained=checkpoint, 
            backbone=dict(
               type='SSL_FCCDN',
               num_band=3,
               os=16,
               use_se=False, 
               dilation_list = [1, 1, 1, 1],
               stride_list = [2, 2, 2, 2],
               pool_list = [True, True, True, True],
               channel_list = [256, 128, 64, 32],
            ),
            cd_decode_head=dict(
                type='SSL_CD_Head',
                in_channels = 3,
                channels=32,
                num_classes=9,
                channel_list = [256, 128, 64, 32],
                pool_list = [True, True, True, True],
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
            )

# optimizer
optimizer=dict(
    type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))