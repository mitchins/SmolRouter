"""
Model resolution strategies for aliasing, priority ordering, and request resolution.

This module implements different strategies for resolving client model requests
to actual provider models, supporting aliases, priority ordering, and fallback logic.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .interfaces import IModelStrategy, ModelInfo, ModelResolution

logger = logging.getLogger(__name__)


@dataclass
class AliasRule:
    """Configuration for model aliasing"""
    pattern: str  # Model pattern to match (can be regex if starts/ends with /)
    target: str   # Target model name or pattern
    priority: int = 0  # Priority for resolution (lower = higher priority)
    
    def matches(self, model_name: str) -> bool:
        """Check if this rule matches a model name"""
        if self.pattern.startswith('/') and self.pattern.endswith('/'):
            # Regex pattern
            pattern = self.pattern[1:-1]
            return bool(re.match(pattern, model_name))
        else:
            # Exact match
            return model_name == self.pattern
    
    def apply(self, model_name: str) -> str:
        """Apply this rule to transform a model name"""
        if self.pattern.startswith('/') and self.pattern.endswith('/'):
            # Regex substitution
            pattern = self.pattern[1:-1]
            match = re.match(pattern, model_name)
            if match:
                return match.expand(self.target)
        
        # Simple replacement
        return self.target


class SmartModelStrategy(IModelStrategy):
    """
    Smart model resolution strategy with advanced features:
    - Model aliasing and transformations
    - Provider priority ordering
    - Fully qualified name resolution (e.g., "llama3-70b [fast-kitten]")
    - Fallback resolution with ranking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.aliases = self._load_alias_rules()
        self.provider_priorities = self._load_provider_priorities()
        self.default_models = self._load_default_models()
    
    def _load_alias_rules(self) -> List[AliasRule]:
        """Load alias rules from configuration"""
        rules = []
        
        # Load from legacy MODEL_MAP format
        model_map = self.config.get('model_map', {})
        for pattern, target in model_map.items():
            rules.append(AliasRule(pattern=pattern, target=target, priority=0))
        
        # Load from new aliases format
        aliases_config = self.config.get('aliases', {})
        for alias_name, alias_config in aliases_config.items():
            if isinstance(alias_config, str):
                # Simple string alias
                rules.append(AliasRule(pattern=alias_name, target=alias_config, priority=0))
            elif isinstance(alias_config, dict):
                # Complex alias with instances (handled by routing layer)
                # For now, just create a simple rule
                target = alias_config.get('target', alias_name)
                priority = alias_config.get('priority', 0)
                rules.append(AliasRule(pattern=alias_name, target=target, priority=priority))
        
        # Sort by priority
        rules.sort(key=lambda r: r.priority)
        
        logger.info(f"Loaded {len(rules)} alias rules")
        return rules
    
    def _load_provider_priorities(self) -> Dict[str, int]:
        """Load provider priority ordering from configuration"""
        priorities = {}
        
        # Load from servers configuration
        servers = self.config.get('servers', {})
        for i, (name, _) in enumerate(servers.items()):
            priorities[name] = i  # Earlier in config = higher priority
        
        # Load explicit priorities
        provider_priorities = self.config.get('provider_priorities', {})
        priorities.update(provider_priorities)
        
        logger.debug(f"Provider priorities: {priorities}")
        return priorities
    
    def _load_default_models(self) -> Dict[str, str]:
        """Load default model mappings"""
        return self.config.get('default_models', {})
    
    async def resolve_model_request(self, requested_model: str, 
                                  available_models: List[ModelInfo]) -> Optional[ModelInfo]:
        """
        Resolve a client's model request to an actual model.
        
        Resolution order:
        1. Check for fully qualified name (e.g., "llama3-70b [fast-kitten]")
        2. Apply alias transformations
        3. Find exact matches
        4. Find partial matches with provider priority
        5. Fallback to similar models
        """
        resolution_path = [f"Resolving request: {requested_model}"]
        
        # Step 1: Check for fully qualified name
        fq_model = self._parse_fully_qualified_name(requested_model)
        if fq_model:
            model_name, provider_id = fq_model
            resolution_path.append(f"Parsed FQ name: {model_name}@{provider_id}")
            
            # Find exact match with specific provider
            for model in available_models:
                if model.provider_id == provider_id and model.matches_request(model_name):
                    resolution_path.append(f"Found FQ match: {model.id}")
                    return ModelResolution(
                        model=model,
                        resolved_from=requested_model,
                        resolution_path=resolution_path
                    ).model
        
        # Step 2: Apply alias transformations
        transformed_name = await self._apply_alias_transformations(requested_model)
        if transformed_name != requested_model:
            resolution_path.append(f"Applied alias: {requested_model} -> {transformed_name}")
            requested_model = transformed_name
        
        # Step 3: Find exact matches
        exact_matches = [
            model for model in available_models 
            if model.matches_request(requested_model)
        ]
        
        if exact_matches:
            resolution_path.append(f"Found {len(exact_matches)} exact matches")
            # Sort by provider priority
            sorted_matches = self._sort_by_provider_priority(exact_matches)
            selected = sorted_matches[0]
            resolution_path.append(f"Selected by priority: {selected.id}")
            return ModelResolution(
                model=selected,
                resolved_from=requested_model,
                resolution_path=resolution_path
            ).model
        
        # Step 4: Find partial matches
        partial_matches = []
        for model in available_models:
            if (requested_model.lower() in model.name.lower() or 
                any(requested_model.lower() in alias.lower() for alias in model.aliases)):
                partial_matches.append(model)
        
        if partial_matches:
            resolution_path.append(f"Found {len(partial_matches)} partial matches")
            sorted_matches = self._sort_by_provider_priority(partial_matches)
            selected = sorted_matches[0]
            resolution_path.append(f"Selected partial match: {selected.id}")
            return ModelResolution(
                model=selected,
                resolved_from=requested_model,
                fallback_used=True,
                resolution_path=resolution_path
            ).model
        
        # Step 5: No matches found
        resolution_path.append("No matches found")
        logger.warning(f"Could not resolve model request: {requested_model}")
        logger.debug(f"Resolution path: {' -> '.join(resolution_path)}")
        
        return None
    
    def _parse_fully_qualified_name(self, requested_model: str) -> Optional[Tuple[str, str]]:
        """
        Parse fully qualified model names like 'llama3-70b [fast-kitten]'

        Returns:
            Tuple of (model_name, provider_id) or None if not FQ format
        """
        # Use string methods to avoid ReDoS vulnerability
        model_str = requested_model.strip()

        # Check if it has the expected format
        if not model_str.endswith(']') or '[' not in model_str:
            return None

        # Find the last '[' to handle model names that might contain '['
        bracket_idx = model_str.rfind('[')
        if bracket_idx == -1:
            return None

        model_name = model_str[:bracket_idx].strip()
        provider_id = model_str[bracket_idx+1:-1].strip()

        # Validate that we have both parts
        if model_name and provider_id:
            return model_name, provider_id

        return None
    
    async def _apply_alias_transformations(self, model_name: str) -> str:
        """Apply alias transformations to a model name"""
        for rule in self.aliases:
            if rule.matches(model_name):
                transformed = rule.apply(model_name)
                logger.debug(f"Applied alias rule: {model_name} -> {transformed}")
                return transformed
        
        return model_name
    
    def _sort_by_provider_priority(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Sort models by provider priority"""
        def get_priority(model: ModelInfo) -> int:
            return self.provider_priorities.get(model.provider_id, 999)
        
        return sorted(models, key=get_priority)
    
    async def apply_aliases(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Apply alias transformations to model list"""
        # For now, return models as-is since aliases are applied during resolution
        # Future enhancement: could add display name transformations here
        return models
    
    async def get_model_priority_order(self, model_name: str) -> List[str]:
        """Get priority order of providers for a given model name"""
        # Return providers sorted by priority
        providers = list(self.provider_priorities.keys())
        providers.sort(key=lambda p: self.provider_priorities.get(p, 999))
        return providers


class SimpleModelStrategy(IModelStrategy):
    """Simple model resolution strategy with basic aliasing"""
    
    def __init__(self, aliases: Dict[str, str] = None):
        self.aliases = aliases or {}
    
    async def resolve_model_request(self, requested_model: str, 
                                  available_models: List[ModelInfo]) -> Optional[ModelInfo]:
        """Simple resolution with exact matching and basic aliases"""
        # Apply aliases first
        resolved_name = self.aliases.get(requested_model, requested_model)
        
        # Find exact match
        for model in available_models:
            if model.matches_request(resolved_name):
                return model
        
        return None
    
    async def apply_aliases(self, models: List[ModelInfo]) -> List[ModelInfo]:
        """Apply alias transformations to model list"""
        return models
    
    async def get_model_priority_order(self, model_name: str) -> List[str]:
        """Return empty priority order (no prioritization)"""
        return []


class StrategyFactory:
    """Factory for creating model strategies"""
    
    @classmethod
    def create_strategy(cls, strategy_type: str = "smart", config: Dict[str, Any] = None) -> IModelStrategy:
        """Create a model strategy instance"""
        if strategy_type == "smart":
            return SmartModelStrategy(config)
        elif strategy_type == "simple":
            aliases = config.get('model_map', {}) if config else {}
            return SimpleModelStrategy(aliases)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")