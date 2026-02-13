from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from HSINerf.HSI_dataparser import HSIDataParserConfig

# Register the HSI dataparser
hsi_dataparser = DataParserSpecification(config=HSIDataParserConfig())