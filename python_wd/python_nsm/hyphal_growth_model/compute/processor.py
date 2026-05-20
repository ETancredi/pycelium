# compute/processor.py

# Imports
from core.point import MPoint # 3D point/vector
from tropisms.field_finder import FieldFinder # Interface for field sources
from dataclasses import dataclass # Used for defining simple data containers
from typing import List # Type hint for lists
from concurrent.futures import ThreadPoolExecutor # Parallel processing

# Simple data structure representing a computation request
@dataclass
class Request:
    point: MPoint # Location where field should be computed
    exclude_ids: List[int] = None # Optional list of source IDs to ignore

# Simple data structure representing the result of a field computation 
@dataclass
class Response:
    field: float # Scalar field value at the point
    gradient: MPoint # Vector gradient of the field at the point

# Processor class for running parallelised field computations
class Processor:
    def __init__(self, field_sources: List[FieldFinder], threads: int = 4):
        self.sources = field_sources # All field sources to be used
        self.threads = threads # No. threads to use for parallel execution

    # Public method to process a batch of Request objects in parallel
    def process_requests(self, requests: List[Request]) -> List[Response]:
        """Run field & gradient computations for each request point."""
        # Use thread pool to process requests concurrently
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            results = list(executor.map(self._handle_request, requests)) # Map each request to a result
        return results # Return list of computed Response objects

    # Internal method for processing single Requests and returning a Response
    def _handle_request(self, req: Request) -> Response:
        total_field = 0.0 # Accumulator for scalar field value
        total_grad = MPoint(0, 0, 0) # Accumulator for gradient vector

        # Loop through all sources to compute contibutions
        for source in self.sources:
            if req.exclude_ids and source.get_id() in req.exclude_ids: # Skip excluded sources
                continue
            total_field += source.find_field(req.point) # Add field value
            grad = source.gradient(req.point) # Compute gradient from source
            total_grad.add(grad) # Accumulate gradients

        # Return total field and normalised gradient vector
        return Response(field=total_field, gradient=total_grad.normalize())
