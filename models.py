# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import List, Optional

class KullaniciProfil(BaseModel):
    id: str
    isim: str
    din: str = "İslam"
    mezhep: str = "Genel"
    ilgi_alanlari: List[str] = Field(default_factory=list) 
    derinlik_seviyesi: str = "Özet ve Pratik"
    dil: str = "tr"
    ses_tercihi: str = "Erkek"

class SoruIstegi(BaseModel):
    user_id: str
    soru: str
    gecmis: List[dict] = []