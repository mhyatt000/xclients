
# webpolicy API guide


### BasePolicy

```python
from webpolicy.base_policy import BasePolicy

class BasePolicy:
    def step(self, obs: dict) -> dict: ...
    def reset(self, *args, **kwargs) -> None: ...
```

### Requirements

* `step(obs: dict) -> dict`

  * Input: JSON-serializable `dict`.
  * Output: JSON-serializable `dict`.
* `reset(*args, **kwargs) -> None`

  * Optional. Called via API to clear state.

---

## 2. Server API

### Python

```python
from webpolicy.server import Server

Server(
    policy: BasePolicy,
    host: str = "0.0.0.0",
    port: int = 8000,
).serve() -> None
```

* `policy`: instance of `BasePolicy` (or subclass).
* `host`: interface to bind.
* `port`: TCP port.
* `serve()`: blocking HTTP server.

## 3. Client API

```python
from webpolicy.client import Client

class Client:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000): ...
    def step(self, obs: dict) -> dict: ...
    def get_server_metadata(self) -> dict: ...
    def reset(self, *args, **kwargs) -> dict | None: ...
```

when implementing a custom server/client,
write a child of `BasePolicy` for the server side, and use Server as is
 On the client side, use Client as is.
