from torch import Tensor, nn


class Flatten(nn.Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Examples::
        >>> input = torch.randn(32, 1, 5, 5)
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
        >>> output = m(input)
        >>> output.size()
        torch.Size([32, 288])
    """
    __constants__ = ["start_dim", "end_dim"]
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        return input.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return "start_dim={}, end_dim={}".format(self.start_dim, self.end_dim)
