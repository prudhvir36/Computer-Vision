import os, sys
import platform
from pathlib import Path

from utils.log import Log
from utils import ret_values
from utils.common_defs import method_header

from meghnad.core.cv.img_clf.cfg.img_clf_config import ImgClfConfig

log = Log()

_config = ImgClfConfig()
_sync_config = _config.get('sync')

@method_header(description='''
               Get meghnad repos's directories''',
               returns='''
               Path to meghnad external repos.''')
def get_meghnad_repo_dir(path) -> Path:
    file_path = Path(os.path.abspath(__file__))
    return file_path.parents[5] / path


@method_header(description='''Get sync directory from config.
''', returns='''Sync dir''')
def get_sync_dir():
    os_name = platform.system().lower()
    if 'linux' in os_name:
        if _sync_config['method'] == 'S3':
            from connectors.aws.s3.config import s3_config_linux

            config = s3_config_linux().get_s3_configs()
            sync_dir = '~/' + Path(config['mount_folder_name'])
        elif _sync_config['method'] == 'ADL':
            from connectors.azure.adl.config import adl_config

            config = adl_config().get_adl_configs_linux()
            sync_dir = '~/' + Path(config['containerName'])
        else:
            pass
    elif 'windows' in os_name:
        if _sync_config['method'] == 'S3':
            from connectors.aws.s3.config import s3_config_windows
            config = s3_config_windows().get_s3_configs()
            sync_dir = config['drive_name'] + ':/' + config['bucket_name']
        elif _sync_config['method'] == 'ADL':
            from connectors.azure.adl.config import adl_config

            config = adl_config().get_adl_configs_windows()
            sync_dir = Path(config['drive_name'] + ':/') + config['blob_name']
        else:
            pass
    else:
        log.ERROR(sys._getframe().f_lineno,
                  __file__, __name__,
                  f'Not supported OS: {os_name}')
        return ret_values.IXO_RET_NOT_SUPPORTED
        #raise ValueError(f'Not supported OS: {os_name}')
    return sync_dir
