class ProviderError(RuntimeError):
    """A call failed. `retryable` decides whether a fallback provider is tried.

    Availability failures — a connection error, a rate limit, an overload, a
    refusal — are retryable. A malformed payload or a missing credential is
    not: the model answered, just unusably, and trying a different provider
    would be shopping for one whose output happens to be usable rather than
    surfacing that this specific answer cannot be trusted.
    """

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def retryable_status(status_code: int) -> bool:
    """Whether an HTTP status is worth asking a different provider about.

    Lives here, not in either wire adapter: the two SDKs raise different
    exception types but the policy is the same, and two copies of it are two
    things that can drift into disagreeing about what an outage looks like.
    """
    return status_code in (408, 409, 429) or status_code >= 500
