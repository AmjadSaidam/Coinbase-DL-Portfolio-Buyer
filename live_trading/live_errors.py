"""
constructors for catching errors 
"""

class CoinLoginError(Exception):
    pass

class CoinDataError(Exception): 
    pass

class CoinOrderError(Exception):
    """raise order error when HTTP does not raise error but repsone.success is False"""
    def __init__(self,
                 product_id: str,
                 side: str,
                 error_repsonse):
        self.product_id = product_id
        self.side = side
        # ref error_repsonse attributes using getattr() to avoid AttributeError()
        self.error = getattr(error_repsonse, 'error', None)
        self.message = getattr(error_repsonse, 'message', None)
        self.details = getattr(error_repsonse, 'error_details', None)

    def __str__(self):
        return f"side {self.side} product_id {self.product_id} rejected: {self.error} - {self.message} ({self.details})"
    