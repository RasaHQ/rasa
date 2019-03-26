import aiohttp
from typing import Any, Optional, Text, Dict
from rasa.constants import DEFAULT_REQUEST_TIMEOUT


def configure_colored_logging(loglevel):
    import coloredlogs
    field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
    field_styles['asctime'] = {}
    level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
    level_styles['debug'] = {}
    coloredlogs.install(
        level=loglevel,
        use_chroot=False,
        fmt='%(asctime)s %(levelname)-8s %(name)s  - %(message)s',
        level_styles=level_styles,
        field_styles=field_styles)


def concat_url(base: Text, subpath: Optional[Text]) -> Text:
    """Append a subpath to a base url.

    Strips leading slashes from the subpath if necessary. This behaves
    differently than `urlparse.urljoin` and will not treat the subpath
    as a base url if it starts with `/` but will always append it to the
    `base`."""

    if not subpath:
        return base

    url = base
    if not base.endswith("/"):
        url += "/"
    if subpath.startswith("/"):
        subpath = subpath[1:]
    return url + subpath


class EndpointConfig(object):
    """Configuration for an external HTTP endpoint."""

    def __init__(self,
                 url: Text = None,
                 params: Dict[Text, Any] = None,
                 headers: Dict[Text, Any] = None,
                 basic_auth: Dict[Text, Text] = None,
                 token: Optional[Text] = None,
                 token_name: Text = "token",
                 **kwargs):
        self.url = url
        self.params = params if params else {}
        self.headers = headers if headers else {}
        self.basic_auth = basic_auth
        self.token = token
        self.token_name = token_name
        self.type = kwargs.pop('store_type', kwargs.pop('type', None))
        self.kwargs = kwargs

    def session(self):
        # create authentication parameters
        if self.basic_auth:
            auth = aiohttp.BasicAuth(self.basic_auth["username"],
                                     self.basic_auth["password"])
        else:
            auth = None

        return aiohttp.ClientSession(
            headers=self.headers,
            auth=auth,
            timeout=aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT),
        )

    def combine_parameters(self, kwargs=None):
        # construct GET parameters
        params = self.params.copy()

        # set the authentication token if present
        if self.token:
            params[self.token_name] = self.token

        if kwargs and "params" in kwargs:
            params.update(kwargs["params"])
            del kwargs["params"]
        return params

    async def request(self,
                      method: Text = "post",
                      subpath: Optional[Text] = None,
                      content_type: Optional[Text] = "application/json",
                      **kwargs: Any
                      ):
        """Send a HTTP request to the endpoint.

        All additional arguments will get passed through
        to aiohttp's `session.request`."""

        # create the appropriate headers
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type

        if "headers" in kwargs:
            headers.update(kwargs["headers"])
            del kwargs["headers"]

        url = concat_url(self.url, subpath)
        async with self.session() as session:
            async with session.request(
                    method,
                    url,
                    headers=headers,
                    params=self.combine_parameters(kwargs),
                    **kwargs) as resp:
                if resp.status >= 400:
                    raise ClientResponseError(resp.status,
                                              resp.reason,
                                              await resp.content.read())
                return await resp.json()

    @classmethod
    def from_dict(cls, data):
        return EndpointConfig(**data)

    def __eq__(self, other):
        if isinstance(self, type(other)):
            return (other.url == self.url and
                    other.params == self.params and
                    other.headers == self.headers and
                    other.basic_auth == self.basic_auth and
                    other.token == self.token and
                    other.token_name == self.token_name)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class ClientResponseError(aiohttp.ClientError):
    def __init__(self, status, message, text):
        self.status = status
        self.message = message
        self.text = text
        super().__init__("{}, {}, body='{}'".format(status, message, text))
