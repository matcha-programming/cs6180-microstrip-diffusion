"""Source-derived architecture descriptions; no PyTorch or checkpoint loading.

Shapes include symbolic batch B and point count P. Learned-module counts include
Conv2d, explicit Linear, and MultiheadAttention once, with its projections grouped.
Normalization, activation, reshape, interpolation, Fourier buffers and global
output bias are excluded from layer counts, but included in parameter totals.
"""
import hashlib
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {
    'experiments/remaining/models.py': '20f12aa865ea5df055c579c30fd1669486b73f0ac1e31917cb76fb3d755a1172',
    'experiments/unet_regression/paired_model.py': '798c3755c463f4bd1b92cd9d78e498374d2f1bcc4abc936aa88d310578fc6f9a',
    'experiments/deeponet_plain/train.py': '32ca458eb54f3060fd9abf99625f6c7d9bf2e746b19250333d423dc3568e25b4',
    'experiments/remaining/fourier.py': 'e6ba5eeeb3471590b80a49bb018222e61a41a4f788aee1f93266895526f85860',
    'experiments/pca_gp/staff_pca_gp.py': '4cf231c86d4270a0a3a6662655e094eabef5d5a409982b183920026a83ae6014',
    'experiments-round-2/support.py': '595183e9f791d42a428ca20a4501e191f9199e11196c7d5efae430757d1c5e3d',
}
COUNT_CONVENTION = (
    'Learned modules count Conv2d, explicit Linear and MultiheadAttention once; '
    'the attention projections are grouped and not double-counted. GroupNorm, '
    'activations, interpolation, reshape, fixed Fourier projection and global '
    'output bias are excluded. This is a module count, not sequential network depth. '
    'The detail index identifies a module or operation, not runtime execution order.'
)


def shape(*dimensions):
    return ' x '.join(map(str, dimensions))


def unet_profile(base, channels):
    """Mirror the four-level ConditionalUNet topology in the verified source."""
    profile = f'UNet-{channels}ch-b{base}'
    rows = []
    parameters = 0

    def add(name, operation, inputs, outputs, weights, biases=0):
        nonlocal parameters
        weight_shapes = weights if isinstance(weights, list) else [weights]
        parameters += sum(math.prod(w) for w in weight_shapes) + biases
        rows.append(dict(index=len(rows) + 1, profile=profile, name=name,
                         operation=operation, input_shape=inputs, output_shape=outputs,
                         weight_shape='; '.join(shape(*w) for w in weight_shapes)))

    def linear(name, cin, cout):
        add(name, 'Linear', shape('B', cin), shape('B', cout), (cout, cin), cout)

    def conv(name, cin, cout, h, w, kernel=3, down=False):
        add(name, f'Conv2d {kernel}x{kernel}' + (' stride 2' if down else ''),
            shape('B', cin, h, w), shape('B', cout, h // 2 if down else h, w // 2 if down else w),
            (cout, cin, kernel, kernel), cout)

    embedding = 4 * base

    def block(name, cin, cout, h, w):
        nonlocal parameters
        # GroupNorm has a scale and bias per channel in each of two norms.
        parameters += 2 * cin + 2 * cout
        conv(name + '.conv1', cin, cout, h, w)
        linear(name + '.embed', embedding, cout)
        conv(name + '.conv2', cout, cout, h, w)
        if cin != cout:
            conv(name + '.skip', cin, cout, h, w, kernel=1)

    linear('time_mlp.0', 128, embedding)
    linear('time_mlp.2', embedding, embedding)
    linear('design_mlp.0', 4, embedding)
    linear('design_mlp.2', embedding, embedding)
    conv('input', channels, base, 64, 128)
    widths = [base, 2 * base, 4 * base, 4 * base]
    current, h, w = base, 64, 128
    for i, width in enumerate(widths):
        block(f'down.{i}.0', current, width, h, w)
        block(f'down.{i}.1', width, width, h, w)
        current = width
        if i < 3:
            conv(f'downsample.{i}', width, width, h, w, down=True)
            h, w = h // 2, w // 2
    block('mid1', current, current, h, w)
    parameters += 2 * current  # Attention GroupNorm.
    add('attn.attn', 'MultiheadAttention, 4 heads', shape('B', h * w, current),
        shape('B', h * w, current), [(3 * current, current), (current, current)], 4 * current)
    block('mid2', current, current, h, w)
    for i, width in enumerate(reversed(widths)):
        block(f'up.{i}.0', current + width, width, h, w)
        block(f'up.{i}.1', width, width, h, w)
        current = width
        if i < 3:
            h, w = h * 2, w * 2  # Nearest-neighbor interpolation precedes convolution.
            next_width = widths[2 - i]
            conv(f'upsample.{i}', current, next_width, h, w)
            current = next_width
    parameters += 2 * base  # Output GroupNorm.
    conv('output.2', base, 2, 64, 128)
    stages = [shape(width, 64 // 2 ** i, 128 // 2 ** i) for i, width in enumerate(widths)]
    summary = 'Encoder ' + ' -> '.join(stages) + '; bottleneck ' + stages[-1] + '; decoder ' + ' -> '.join(reversed(stages)) + '; head 2 x 64 x 128'
    return dict(profile=profile, learned_modules=len(rows), layer_breakdown='50 Conv2d + 22 Linear + 1 attention; 18 residual blocks; 4 encoder/4 decoder levels',
                input_shape=shape('B', channels, 64, 128) + '; design B x 4; time B',
                output_shape=shape('B', 2, 64, 128), layer_shapes=summary,
                calculated_parameters=parameters, layers=rows)


def deeponet_profile(width, rank, fourier):
    profile = f"{'Fourier' if fourier else 'Plain'}-DeepONet-w{width}-r{rank}"
    rows, parameters = [], 2  # The two output bias parameters are not layers.
    for branch, first in [('branch', 4), ('trunk', 256 if fourier else 2)]:
        dimensions = [first, width, width, 2 * rank]
        for j, (cin, cout) in enumerate(zip(dimensions, dimensions[1:])):
            parameters += cin * cout + cout
            axes = ['B'] if branch == 'branch' else ['B', 'P']
            rows.append(dict(index=len(rows) + 1, profile=profile, name=f'{branch}.{2*j}', operation='Linear',
                             input_shape=shape(*axes, cin), output_shape=shape(*axes, cout), weight_shape=shape(cout, cin)))
    if fourier:
        rows.append(dict(index=None, profile=profile, name='Fixed Fourier features', operation='sin/cos(2 pi B coords/8)',
                         input_shape=shape('B', 'P', 2), output_shape=shape('B', 'P', 256), weight_shape='128 x 2 (fixed buffer)'))
    trunk = ('2 -> fixed Fourier 256 -> ' if fourier else '2 -> ') + f'{width} -> {width} -> {2*rank}'
    return dict(profile=profile, learned_modules=6, layer_breakdown='6 Linear: 3 branch + 3 trunk; SiLU after the first 2 layers of each MLP',
                input_shape='design B x 4; coordinates B x P x 2', output_shape='B x 2 x P; full grid B x 2 x 64 x 128',
                layer_shapes=f'Branch 4 -> {width} -> {width} -> {2*rank}; trunk {trunk}; reshape both to 2 x {rank}; rank-{rank} dot product per channel',
                calculated_parameters=parameters, layers=rows)


def pca_profile(k):
    profile = f'PCA{k}-GP'
    rows = [
        dict(index=None, profile=profile, name='Independent coefficient predictors', operation=f'{k} ARD RBF Gaussian processes', input_shape='B x 4', output_shape=shape('B', k), weight_shape='n.a. (kernel model)'),
        dict(index=None, profile=profile, name='Coefficient unstandardization', operation='Scale and offset', input_shape=shape('B', k), output_shape=shape('B', k), weight_shape=shape(k) + ' scale and offset'),
        dict(index=None, profile=profile, name='Inverse PCA and reshape', operation='PCA reconstruction', input_shape=shape('B', k), output_shape='B x 2 x 64 x 128', weight_shape=shape(k, 16384) + '; mean 16384'),
    ]
    return dict(profile=profile, learned_modules=None, layer_breakdown=f'Non-neural: PCA rank {k} and {k} independent GPs',
                input_shape='B x 4 design parameters', output_shape='B x 2 x 64 x 128',
                layer_shapes=f'4 inputs -> {k} predicted PCA coefficients -> inverse PCA {k} -> 16384 -> 2 x 64 x 128',
                calculated_parameters=None, layers=rows)


def architecture_for(plan_id, config):
    """Return supported topology and source evidence, or explicit unavailable data."""
    sources = []
    if plan_id == '1' and 'k' in config:
        details = pca_profile(config['k'])
        sources = ['experiments/pca_gp/staff_pca_gp.py']
    elif plan_id in ('2', '3') and all(k in config for k in ('width', 'rank')):
        details = deeponet_profile(config['width'], config['rank'], plan_id == '3')
        sources = ['experiments/remaining/fourier.py' if plan_id == '3' else 'experiments/deeponet_plain/train.py']
    elif plan_id in ('4', '4R', '6', 'R2-A', 'R2-B', 'R2-C', 'R2-D') and 'base' in config:
        paired = plan_id == '4R' or plan_id.startswith('R2-') and config.get('backbone') != 'original' or bool(config.get('position_conditioning'))
        channels = 4 if paired or plan_id == '6' else 2
        details = unet_profile(config['base'], channels)
        sources = ['experiments/remaining/models.py']
        if paired:
            sources.append('experiments-round-2/support.py' if plan_id.startswith('R2-') else 'experiments/unet_regression/paired_model.py')
        semantics = 'two zero field planes + two fixed index planes; time zero' if plan_id == '4R' else ('two residual/noisy planes + two frozen regressor planes' if plan_id == '6' else ('two field/noisy planes + two fixed index planes' if paired else 'two field/noisy planes'))
        details['input_shape'] += '; ' + semantics
    else:
        return dict(profile='Unavailable', learned_modules=None, layer_breakdown='Architecture configuration unavailable', input_shape=None, output_shape=None, layer_shapes=None, calculated_parameters=None, layers=[], sources=[])
    for source in sources:
        digest = hashlib.sha256((REPO / source).read_bytes()).hexdigest()
        if digest != SOURCE_HASHES[source]:
            raise ValueError(f'Architecture source changed: {source}; review model_architectures.py before publishing inferred layer shapes')
    recorded = config.get('parameters')
    if recorded is not None and details['calculated_parameters'] != recorded:
        raise ValueError(f"Architecture parameter count {details['calculated_parameters']} disagrees with saved configuration {recorded}")
    details['sources'] = sources
    return details
