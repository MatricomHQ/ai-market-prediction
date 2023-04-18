"""
Author: Michael B1rown
Github: @slapglif
"""

import logging
from typing import Dict, List

logger = logging.getLogger('app')

from config import Config

from pydantic import BaseModel, Field


class Registry(BaseModel):
    routes: List[str] = Field(default=[
        "broker"
    ])
