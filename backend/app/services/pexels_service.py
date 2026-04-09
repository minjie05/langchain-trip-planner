"""Pexels图片服务"""

import requests
from typing import List, Optional
from ..config import get_settings

class PexelsService:
    """Pexels图片服务类"""

    def __init__(self):
        """初始化服务"""
        settings = get_settings()
        self.api_key = settings.pexels_api_key
        self.base_url = "https://api.pexels.com/v1"

    def search_photos(self, query: str, per_page: int = 5) -> List[dict]:
        """
        搜索图片

        Args:
            query: 搜索关键词
            per_page: 每页数量

        Returns:
            图片列表
        """
        try:
            url = f"{self.base_url}/search"
            headers = {"Authorization": self.api_key}
            params = {
                "query": query,
                "per_page": per_page
            }

            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            photos = data.get("photos", [])

            # 提取图片URL
            results = []
            for photo in photos:
                results.append({
                    "id": str(photo.get("id")),
                    "url": photo.get("src", {}).get("large"),
                    "thumb": photo.get("src", {}).get("medium"),
                    "description": photo.get("alt"),
                    "photographer": photo.get("photographer")
                })

            return results

        except Exception as e:
            print(f"❌ Pexels搜索失败: {str(e)}")
            return []

    def get_photo_url(self, query: str) -> Optional[str]:
        """
        获取单张图片URL

        Args:
            query: 搜索关键词

        Returns:
            图片URL
        """
        photos = self.search_photos(query, per_page=1)
        if photos:
            return photos[0].get("url")
        return None


# 全局服务实例
_pexels_service = None


def get_pexels_service() -> PexelsService:
    """获取Pexels服务实例(单例模式)"""
    global _pexels_service

    if _pexels_service is None:
        _pexels_service = PexelsService()

    return _pexels_service
