import subprocess

from path import Path

from charms.reactive import when, when_not, is_state
from charms.reactive import set_state, remove_state
from charmhelpers.core import hookenv
from charmhelpers import fetch


@when_not('tensorflow.installed')
def first_install():
    set_state('tensorflow.install')


@when('config.changed.gpu_support')
def reinstall():
    set_state('tensorflow.install')
    remove_state('config.changed.gpu_support')


@when('tensorflow.install')
@when_not('juju-1.0')
def install():
    # this handler is separated from first_install and reinstall to ensure
    # it only gets called once, even if both of the above trigger
    remove_state('tensorflow.install')
    gpu_support = is_state('config.set.gpu_support')
    if is_state('tensorflow.installed'):
        hookenv.status_set('maintenance', 'reinstalling tensorflow')
    else:
        hookenv.status_set('maintenance', 'installing tensorflow')
    try:
        arch = subprocess.check_output(['uname', '-p']).decode('utf8').strip()
        resource = 'tensorflow-{}'.format(arch)
        if gpu_support:
            resource += '-gpu'
        tensorflow_file = hookenv.resource_get(resource)
        if not tensorflow_file:
            hookenv.status_set('blocked',
                               'unable to fetch tensorflow resource')
            return
        if gpu_support:
            cudnn_file = hookenv.resource_get(resource)
            if not cudnn_file or Path(cudnn_file).size == 0:
                hookenv.status_set(
                    'blocked',
                    'unable to fetch cudnn resource for gpu support')
                return
    except NotImplementedError:
        hookenv.status_set('blocked', 'requires juju 2.0+')
        set_state('juju-1.0')  # don't bother trying to install again
        return

    if is_state('tensorflow.installed'):
        subprocess.check_call(['pip3', 'uninstall', 'tensorflow'])
    subprocess.check_call(['pip3', 'install', tensorflow_file])
    if gpu_support:
        # this probably doesn't work, but the cuda blobs are huge
        fetch.apt_install([
            'libcuda-361',
            'nvidia-cuda-toolkit',
        ])
        toolkit = Path('/usr/lib/nvidia-cuda-toolkit')
        extracted = Path(fetch.install_remote('file://' + cudnn_file)) / 'cuda'
        (extracted / 'include' / 'cudnn.h').copy(toolkit / 'include')
        for filename in (extracted / 'lib64').files('libcudnn*'):
            filename.copy(toolkit / 'lib64')
        subprocess.check_call(['chmod', 'a+r',
                               toolkit / 'include/cudnn.h'] +
                              (toolkit / 'lib64').files('libcudnn*'))

    set_state('tensorflow.installed')
    hookenv.status_set('active', 'ready')
