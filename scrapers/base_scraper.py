"""
Aerial Leads - Base Scraper

Abstract base class for all web scrapers.
Provides common functionality: requests, retries, rate limiting, error handling.
"""

import time
import requests
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod
import logging
from config.settings import (
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    USER_AGENT,
    USE_PROXY,
    PROXY_URL
)
from config.logging_config import get_logger


class BaseScraper(ABC):
    """
    Base class for all web scrapers

    Provides:
    - HTTP request handling with retries
    - Rate limiting
    - Error handling and logging
    - Proxy support
    - User agent rotation
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.session = self._create_session()
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0

    def _create_session(self) -> requests.Session:
        """Create and configure a requests session"""
        session = requests.Session()

        # Set headers
        session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

        # Set proxy if configured
        if USE_PROXY and PROXY_URL:
            session.proxies = {
                'http': PROXY_URL,
                'https': PROXY_URL
            }
            self.logger.info(f"Using proxy: {PROXY_URL}")

        return session

    def _make_request(
        self,
        url: str,
        method: str = 'GET',
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        json: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: int = REQUEST_TIMEOUT,
        retry_count: int = 0
    ) -> Optional[requests.Response]:
        """
        Make HTTP request with retry logic

        Args:
            url: URL to request
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Form data
            json: JSON data
            headers: Additional headers
            timeout: Request timeout in seconds
            retry_count: Current retry attempt

        Returns:
            Response object or None if failed
        """
        self.request_count += 1

        # Add rate limiting
        if self.request_count > 1:
            time.sleep(REQUEST_DELAY)

        # Merge headers
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)

        try:
            self.logger.debug(f"Making {method} request to {url}")

            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json,
                headers=request_headers,
                timeout=timeout
            )

            # Check for successful status
            if response.status_code == 200:
                self.success_count += 1
                self.logger.debug(f"Request successful: {url}")
                return response

            elif response.status_code == 429:
                # Rate limited
                self.logger.warning(f"Rate limited on {url}, waiting 30 seconds...")
                time.sleep(30)

                if retry_count < MAX_RETRIES:
                    return self._make_request(
                        url, method, params, data, json, headers, timeout, retry_count + 1
                    )

            elif response.status_code in [500, 502, 503, 504]:
                # Server error, retry
                self.logger.warning(f"Server error ({response.status_code}) on {url}")

                if retry_count < MAX_RETRIES:
                    time.sleep(5 * (retry_count + 1))  # Exponential backoff
                    return self._make_request(
                        url, method, params, data, json, headers, timeout, retry_count + 1
                    )

            else:
                self.logger.error(f"Request failed with status {response.status_code}: {url}")
                self.error_count += 1
                return None

        except requests.exceptions.Timeout:
            self.logger.error(f"Request timeout: {url}")
            if retry_count < MAX_RETRIES:
                return self._make_request(
                    url, method, params, data, json, headers, timeout, retry_count + 1
                )
            self.error_count += 1
            return None

        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error: {url} - {str(e)}")
            if retry_count < MAX_RETRIES:
                time.sleep(5 * (retry_count + 1))
                return self._make_request(
                    url, method, params, data, json, headers, timeout, retry_count + 1
                )
            self.error_count += 1
            return None

        except Exception as e:
            self.logger.error(f"Unexpected error during request: {str(e)}", exc_info=True)
            self.error_count += 1
            return None

    def _parse_currency(self, text: str) -> float:
        """
        Parse currency string to float

        Args:
            text: Currency string (e.g., "$12,345.67")

        Returns:
            Float value
        """
        if not text:
            return 0.0

        # Remove currency symbols and commas
        cleaned = text.replace('$', '').replace(',', '').strip()

        try:
            return float(cleaned)
        except ValueError:
            self.logger.warning(f"Could not parse currency: {text}")
            return 0.0

    def _parse_number(self, text: str) -> int:
        """
        Parse number string to int

        Args:
            text: Number string (e.g., "1,234")

        Returns:
            Integer value
        """
        if not text:
            return 0

        # Remove commas
        cleaned = text.replace(',', '').strip()

        try:
            return int(cleaned)
        except ValueError:
            self.logger.warning(f"Could not parse number: {text}")
            return 0

    def get_stats(self) -> Dict[str, int]:
        """Get scraper statistics"""
        return {
            'total_requests': self.request_count,
            'successful_requests': self.success_count,
            'failed_requests': self.error_count,
            'success_rate': round(self.success_count / self.request_count * 100, 2) if self.request_count > 0 else 0
        }

    def reset_stats(self):
        """Reset scraper statistics"""
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0

    @abstractmethod
    def scrape(self, *args, **kwargs) -> Any:
        """
        Main scraping method - must be implemented by subclasses
        """
        pass

    def __del__(self):
        """Clean up session on deletion"""
        if hasattr(self, 'session'):
            self.session.close()
