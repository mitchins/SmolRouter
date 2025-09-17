import re 
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import httpx

logger = logging.getLogger("model-rerouter")

@dataclass
class UpstreamInstance:
    """Represents a single upstream server instance"""
    name: str
    url: str
    model: Optional[str] = None  # Model override for this instance
    
    def __str__(self):
        return f"{self.name}({self.url})"

@dataclass
class ModelAlias:
    """Represents a model alias with multiple upstream instances"""
    name: str
    instances: List[UpstreamInstance]
    
    def __str__(self):
        return f"{self.name} -> [{', '.join(str(i) for i in self.instances)}]"

class SmartRouter:
    """Enhanced routing with model aliases and automatic failover"""
    
    def __init__(self, config: Dict[str, Any], default_upstream: str):
        self.default_upstream = default_upstream
        self.aliases: Dict[str, ModelAlias] = {}
        self.routes: List[Dict[str, Any]] = []
        self._load_config(config)
    
    def _load_config(self, config: Dict[str, Any]):
        """Load routing configuration including aliases"""
        # Load traditional routes
        self.routes = config.get('routes', [])
        
        # Load model aliases
        aliases_config = config.get('aliases', {})
        for alias_name, alias_config in aliases_config.items():
            instances = []
            
            for instance_config in alias_config.get('instances', []):
                if isinstance(instance_config, str):
                    # Simple format: "server/model"
                    if '/' in instance_config:
                        server_name, model = instance_config.split('/', 1)
                    else:
                        server_name, model = instance_config, None
                    
                    # Look up server URL from servers config
                    servers = config.get('servers', {})
                    if server_name in servers:
                        url = servers[server_name]
                        instances.append(UpstreamInstance(server_name, url, model))
                    else:
                        logger.error(f"Unknown server '{server_name}' in alias '{alias_name}'")
                
                elif isinstance(instance_config, dict):
                    # Advanced format with explicit config
                    server_name = instance_config.get('server')
                    url = instance_config.get('url')
                    model = instance_config.get('model')
                    
                    if server_name and not url:
                        # Look up URL from servers config
                        servers = config.get('servers', {})
                        url = servers.get(server_name)
                    
                    if url:
                        instances.append(UpstreamInstance(server_name or url, url, model))
                    else:
                        logger.error(f"Invalid instance config in alias '{alias_name}': {instance_config}")
            
            if instances:
                self.aliases[alias_name] = ModelAlias(alias_name, instances)
                logger.info(f"Loaded alias: {self.aliases[alias_name]}")
        
        logger.info(f"Loaded {len(self.aliases)} model aliases and {len(self.routes)} routes")
    
    def find_route(self, source_host: str, model: str) -> Tuple[List[UpstreamInstance], Optional[str]]:
        """Find routing targets for a request.
        
        Returns:
            Tuple of (instances_to_try, final_model_override)
            instances_to_try is a list of UpstreamInstance objects to try in order
        """
        # First check if model matches an alias
        if model in self.aliases:
            alias = self.aliases[model]
            logger.debug(f"Model '{model}' matched alias with {len(alias.instances)} instances")
            return alias.instances, None
        
        # Check traditional routing rules
        for route in self.routes:
            match_criteria = route.get('match', {})
            route_config = route.get('route', {})
            
            # Check if this route matches
            matches = True
            
            # Check source host match (if specified)
            if 'source_host' in match_criteria:
                expected_host = match_criteria['source_host']
                if source_host != expected_host:
                    matches = False
                    
            # Check model match (if specified) - supports regex
            if matches and 'model' in match_criteria:
                model_pattern = match_criteria['model']
                if model_pattern.startswith('/') and model_pattern.endswith('/'):
                    # Regex pattern
                    pattern = model_pattern[1:-1]  # Remove slashes
                    if not re.search(pattern, model):
                        matches = False
                else:
                    # Exact match
                    if model != model_pattern:
                        matches = False
            
            if matches:
                upstream = route_config.get('upstream')
                model_override = route_config.get('model')
                
                if upstream:
                    logger.debug(f"Route matched: {source_host}/{model} -> {upstream}" +
                               (f" (model: {model_override})" if model_override else ""))
                    instance = UpstreamInstance("route", upstream, model_override)
                    return [instance], model_override
        
        # No specific route found, use default
        logger.debug(f"No specific route found for {source_host}/{model}, using default upstream")
        default_instance = UpstreamInstance("default", self.default_upstream, None)
        return [default_instance], None
    
    async def try_upstream(self, instance: UpstreamInstance, request_payload: Dict[str, Any], 
                          path: str, headers: Dict[str, str], timeout: float) -> Tuple[bool, Any, int]:
        """Try a single upstream instance.
        
        Returns:
            Tuple of (success, response_data_or_error, status_code)
        """
        # Apply model override if specified
        if instance.model:
            request_payload = request_payload.copy()
            request_payload["model"] = instance.model
            logger.debug(f"Using model override '{instance.model}' for {instance.name}")
        
        url = f"{instance.url}{path}"
        logger.debug(f"Trying upstream {instance.name}: {url}")
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=request_payload, headers=headers)
                
                if resp.status_code < 400:
                    # Success
                    try:
                        data = resp.json()
                        logger.debug(f"Upstream {instance.name} succeeded with status {resp.status_code}")
                        return True, data, resp.status_code
                    except Exception as e:
                        logger.error(f"Failed to parse JSON from {instance.name}: {e}")
                        return False, f"JSON parse error: {e}", resp.status_code
                else:
                    # HTTP error
                    try:
                        error_data = resp.json()
                        logger.warning(f"Upstream {instance.name} returned {resp.status_code}: {error_data}")
                        return False, error_data, resp.status_code
                    except Exception:
                        logger.warning(f"Upstream {instance.name} returned {resp.status_code}")
                        return False, f"HTTP {resp.status_code}", resp.status_code
                        
        except httpx.ConnectError as e:
            logger.warning(f"Connection error to {instance.name}: {e}")
            return False, f"Connection error: {e}", 502
        except httpx.TimeoutException as e:
            logger.warning(f"Timeout error to {instance.name}: {e}")
            return False, f"Timeout: {e}", 504
        except Exception as e:
            logger.error(f"Unexpected error with {instance.name}: {e}")
            return False, f"Unexpected error: {e}", 500
    
    async def route_request(self, source_host: str, model: str, request_payload: Dict[str, Any],
                           path: str, headers: Dict[str, str], timeout: float) -> Tuple[Any, int, str]:
        """Route a request with automatic failover.
        
        Returns:
            Tuple of (response_data, status_code, upstream_used)
        """
        instances_to_try, model_override = self.find_route(source_host, model)

        errors = []
        last_status_code = 502  # Default to bad gateway if no attempts made

        for i, instance in enumerate(instances_to_try):
            logger.debug(f"Trying upstream {i+1}/{len(instances_to_try)}: {instance}")

            success, result, status_code = await self.try_upstream(
                instance, request_payload, path, headers, timeout
            )

            # Always track the last status code we received
            last_status_code = status_code

            if success:
                logger.info(f"Request succeeded via {instance} ({status_code})")
                return result, status_code, str(instance)
            else:
                errors.append(f"{instance}: {result}")
                logger.warning(f"Upstream {instance} failed ({status_code}): {result}")

        # All upstreams failed
        logger.error(f"All {len(instances_to_try)} upstreams failed for {model}")

        # Use the last status code we got from actual attempts
        final_status = last_status_code
        
        error_response = {
            "error": "all_upstreams_failed",
            "message": f"All {len(instances_to_try)} configured upstreams failed",
            "details": errors
        }
        
        return error_response, final_status, "none"

# Global router instance
_smart_router: Optional[SmartRouter] = None

def get_smart_router(config: Dict[str, Any], default_upstream: str) -> SmartRouter:
    """Get or create the global smart router instance"""
    global _smart_router
    if _smart_router is None:
        _smart_router = SmartRouter(config, default_upstream)
    return _smart_router

def reload_router_config(config: Dict[str, Any], default_upstream: str):
    """Reload router configuration"""
    global _smart_router
    _smart_router = SmartRouter(config, default_upstream)
    logger.info("Router configuration reloaded")