_base_ = [
    '../_base_/models/swin.py', 
    '../common/fccdn_512x512_40k_my-seg-data.py']
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = None # noqa

model = dict(
    type='SDIEncoderDecoder', 
    pretrained=checkpoint, 
    backbone=dict(
        in_channels=3,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    cd_decode_head=dict(
                type='SSL_CD_Head',
                in_channels = 3,
                channels=96,
                num_classes=9,
                channel_list = [768, 384, 192, 96],
                pool_list = [True, True, True, True],
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

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

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    )
]