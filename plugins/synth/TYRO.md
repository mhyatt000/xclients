
if its a dataclass, then
```python
@dataclass
class Myclass:
    """ Myclass is a dataclass that represents a simple example. """
    name: str # it is better to describe the argument here
    group: int # and here rather than the docstring
```

NOT
```python
@dataclass
class Myclass:
    """ Myclass is a dataclass that represents a simple example.
    Args:
      name: this is wasting space and tokens
      group: dont waste space
    """
    # comment about name
    name: str
    # another comment
    group: int
```

when soliciting args with tyro:

```python
import tyro

def main(cfg: Config):
  pass

if __name__ == '__main__':
    main(tyro.cli(Config))
```

NOT
```python
if __name__ == '__main__':
    tyro.cli(main)
```
