import os
import ssl
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Text, Union

import aiohttp
import structlog
from aiohttp.client_exceptions import ContentTypeError
from sanic.request import Request

from rasa.core.actions.constants import MISSING_DOMAIN_MARKER
from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa.shared.exceptions import FileNotFoundException
from rasa.shared.utils.yaml import read_config_file
from rasa.tracing.constants import TRACING_TYPE_LANGFUSE
from rasa.tracing.exceptions import DuplicateTracingConfigException

structlogger = structlog.get_logger()


@lru_cache(maxsize=10)
def read_endpoint_config(
    filename: Union[str, Path], endpoint_type: Text
) -> Optional["EndpointConfig"]:
    """Read an endpoint configuration file from disk and extract one config."""
    if not filename:
        return None

    try:
        content = read_config_file(filename)

        structlogger.debug(
            "endpoint.read.success",
            filename=os.path.abspath(filename),
            endpoint_type=endpoint_type,
            event_info="Successfully read endpoint configuration file.",
            content=content,
        )

        if content.get(endpoint_type) is None:
            return None

        return EndpointConfig.from_dict(content[endpoint_type])
    except FileNotFoundError:
        structlogger.error(
            "endpoint.read.failed_no_such_file",
            filename=os.path.abspath(filename),
            event_info=(
                "Failed to read endpoint configuration file - the file was not found."
            ),
        )
        return None


@lru_cache(maxsize=10)
def read_backend_tracing_configuration(
    filename: Union[str, Path], endpoint_type: str
) -> Optional["EndpointConfig"]:
    """Read a list of endpoint configurations from a yaml file."""
    if not filename:
        return None

    try:
        content = read_config_file(filename)
    except FileNotFoundError:
        structlogger.error(
            "endpoint.read_backend_tracing_configuration.error",
            filename=os.path.abspath(filename),
            event_info=(
                "Failed to read endpoint configuration file - the file was not found."
            ),
        )
        return None

    structlogger.debug(
        "endpoint.read.success",
        filename=os.path.abspath(filename),
        endpoint_type=endpoint_type,
        event_info="Successfully read endpoint configuration file.",
        content=content,
    )

    if content.get(endpoint_type) is None:
        return None

    config: Optional[Union[List[Dict[Text, Any]], Dict[Text, Any]]] = content.get(
        endpoint_type, None
    )

    if config is None:
        return None

    tracing_configs: List[EndpointConfig] = []
    if isinstance(config, list):
        tracing_configs = [EndpointConfig.from_dict(item) for item in config]
    else:
        tracing_configs = [EndpointConfig.from_dict(config)]

    # remove tracing config for langfuse as it is handled differently
    tracing_configs = [
        tracing_config
        for tracing_config in tracing_configs
        if tracing_config.type != TRACING_TYPE_LANGFUSE
    ]

    # if there are no tracing configs, return None
    if len(tracing_configs) == 0:
        return None

    # if there are multiple tracing configs, raise an error
    if len(tracing_configs) > 1:
        structlogger.error(
            "endpoint.read.multiple_tracing_configs",
            filename=os.path.abspath(filename),
            event_info=(
                "Multiple tracing configs found in the endpoints file, which is not "
                "supported. Only one tracing config is allowed.",
            ),
            tracing_configs=tracing_configs,
        )
        raise DuplicateTracingConfigException(
            f"Multiple tracing configs found in {filename}: {tracing_configs}"
        )

    return tracing_configs[0]


def read_property_config_from_endpoints_file(
    filename: Union[str, Path], property_name: str
) -> Optional[Union[Dict[str, Any], List]]:
    """Read a property from an endpoint configuration file."""
    if not filename:
        return None

    try:
        content = read_config_file(filename)

        if content.get(property_name) is None:
            return None

        return content[property_name]
    except FileNotFoundError:
        structlogger.error(
            "endpoint.read.failed_no_such_file",
            filename=os.path.abspath(filename),
            event_info=(
                "Failed to read endpoint configuration file - the file was not found."
            ),
        )
        return None


def concat_url(base: Text, subpath: Optional[Text]) -> Text:
    """Append a subpath to a base url.

    Strips leading slashes from the subpath if necessary. This behaves
    differently than `urlparse.urljoin` and will not treat the subpath
    as a base url if it starts with `/` but will always append it to the
    `base`.

    Args:
        base: Base URL.
        subpath: Optional path to append to the base URL.

    Returns:
        Concatenated URL with base and subpath.
    """
    if not subpath:
        if base.endswith("/"):
            structlogger.debug(
                "endpoint.concat_url.trailing_slash",
                url=base,
                event_info=(
                    "The URL has a trailing slash. Please make sure the "
                    "target server supports trailing slashes for this endpoint."
                ),
            )
        return base

    url = base
    if not base.endswith("/"):
        url += "/"
    if subpath.startswith("/"):
        subpath = subpath[1:]
    return url + subpath


class EndpointConfig:
    """Configuration for an external HTTP endpoint."""

    def __init__(
        self,
        url: Optional[Text] = None,
        params: Optional[Dict[Text, Any]] = None,
        headers: Optional[Dict[Text, Any]] = None,
        basic_auth: Optional[Dict[Text, Text]] = None,
        token: Optional[Text] = None,
        token_name: Text = "token",
        cafile: Optional[Text] = None,
        actions_module: Optional[Union[Text, ModuleType]] = None,
        **kwargs: Any,
    ) -> None:
        """Creates an `EndpointConfig` instance."""
        self.url = url
        self.params = params or {}
        self.headers = headers or {}
        self.basic_auth = basic_auth or {}
        self.token = token
        self.token_name = token_name
        self.type = kwargs.pop("store_type", kwargs.pop("type", None))
        self.cafile = cafile
        self.actions_module = actions_module
        self.kwargs = kwargs

    def session(self) -> aiohttp.ClientSession:
        """Creates and returns a configured aiohttp client session."""
        # create authentication parameters
        if self.basic_auth:
            auth = aiohttp.BasicAuth(
                self.basic_auth["username"], self.basic_auth["password"]
            )
        else:
            auth = None

        return aiohttp.ClientSession(
            headers=self.headers,
            auth=auth,
            timeout=aiohttp.ClientTimeout(total=DEFAULT_REQUEST_TIMEOUT),
        )

    def combine_parameters(
        self, kwargs: Optional[Dict[Text, Any]] = None
    ) -> Dict[Text, Any]:
        # construct GET parameters
        params = self.params.copy()

        # set the authentication token if present
        if self.token:
            params[self.token_name] = self.token

        if kwargs and "params" in kwargs:
            params.update(kwargs["params"])
            del kwargs["params"]
        return params

    async def request(
        self,
        method: Text = "post",
        subpath: Optional[Text] = None,
        content_type: Optional[Text] = "application/json",
        compress: bool = False,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Send a HTTP request to the endpoint. Return json response, if available.

        All additional arguments will get passed through
        to aiohttp's `session.request`.
        """
        # create the appropriate headers
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type

        if "headers" in kwargs:
            headers.update(kwargs["headers"])
            del kwargs["headers"]

        if self.headers:
            headers.update(self.headers)

        url = concat_url(self.url, subpath)

        sslcontext = None
        if self.cafile:
            try:
                # create a SSL context with the provided CA file
                # and set the minimum TLS version to 1.2
                # Purpose is set to SERVER_AUTH to verify the server's certificate
                sslcontext = ssl.create_default_context(
                    purpose=ssl.Purpose.SERVER_AUTH, cafile=self.cafile
                )
                sslcontext.minimum_version = ssl.TLSVersion.TLSv1_2
            except FileNotFoundError as e:
                raise FileNotFoundException(
                    f"Failed to find certificate file, "
                    f"'{os.path.abspath(self.cafile)}' does not exist."
                ) from e

        async with self.session() as session:
            async with session.request(
                method,
                url,
                headers=headers,
                params=self.combine_parameters(kwargs),
                compress=compress,
                ssl=sslcontext,
                **kwargs,
            ) as response:
                if response.status == 449:
                    # Return a special marker that HTTPCustomActionExecutor can detect
                    # This avoids raising an exception for this expected case
                    return {MISSING_DOMAIN_MARKER: True}

                if response.status >= 400:
                    raise ClientResponseError(
                        response.status,
                        response.reason,
                        await response.content.read(),
                    )
                try:
                    return await response.json()
                except ContentTypeError:
                    return None

    @classmethod
    def from_dict(cls, data: Dict[Text, Any]) -> "EndpointConfig":
        return EndpointConfig(**data)

    def copy(self) -> "EndpointConfig":
        return EndpointConfig(
            self.url,
            self.params,
            self.headers,
            self.basic_auth,
            self.token,
            self.token_name,
            **self.kwargs,
        )

    def to_dict(self) -> Dict[Text, Any]:
        """Convert the endpoint config to a dictionary."""
        data = {
            "url": self.url,
            "params": self.params,
            "headers": self.headers,
            "basic_auth": self.basic_auth,
            "token": self.token,
            "token_name": self.token_name,
            "cafile": self.cafile,
            "actions_module": self.actions_module,
        }
        data.update(self.kwargs)
        return data

    def __eq__(self, other: Any) -> bool:
        if isinstance(self, type(other)):
            return (
                other.url == self.url
                and other.params == self.params
                and other.headers == self.headers
                and other.basic_auth == self.basic_auth
                and other.token == self.token
                and other.token_name == self.token_name
            )
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


class ClientResponseError(aiohttp.ClientError):
    def __init__(self, status: int, message: Text, text: Text) -> None:
        self.status = status
        self.message = message
        self.text = text
        super().__init__(f"{status}, {message}, body='{text}'")


def bool_arg(request: Request, name: Text, default: bool = True) -> bool:
    """Returns a passed boolean argument of the request or a default.

    Checks the `name` parameter of the request if it contains a valid
    boolean value. If not, `default` is returned.

    Args:
        request: Sanic request.
        name: Name of argument.
        default: Default value for `name` argument.

    Returns:
        A bool value if `name` is a valid boolean, `default` otherwise.
    """
    return str(request.args.get(name, default)).lower() == "true"


def float_arg(
    request: Request, key: Text, default: Optional[float] = None
) -> Optional[float]:
    """Returns a passed argument cast as a float or None.

    Checks the `key` parameter of the request if it contains a valid
    float value. If not, `default` is returned.

    Args:
        request: Sanic request.
        key: Name of argument.
        default: Default value for `key` argument.

    Returns:
        A float value if `key` is a valid float, `default` otherwise.
    """
    arg = request.args.get(key, default)

    if arg is default:
        return arg

    try:
        return float(str(arg))
    except (ValueError, TypeError):
        structlogger.warning("endpoint.float_arg.convert_failed", arg=arg, key=key)
        return default


def int_arg(
    request: Request, key: Text, default: Optional[int] = None
) -> Optional[int]:
    """Returns a passed argument cast as an int or None.

    Checks the `key` parameter of the request if it contains a valid
    int value. If not, `default` is returned.

    Args:
        request: Sanic request.
        key: Name of argument.
        default: Default value for `key` argument.

    Returns:
        An int value if `key` is a valid integer, `default` otherwise.
    """
    arg = request.args.get(key, default)

    if arg is default:
        return arg

    try:
        return int(str(arg))
    except (ValueError, TypeError):
        structlogger.warning("endpoint.int_arg.convert_failed", arg=arg, key=key)
        return default
