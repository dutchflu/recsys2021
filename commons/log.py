"""
logging settings
"""

import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(threadName)s][%(levelname)-5.5s]  %(message)s"
                    )
log = logging.getLogger()
