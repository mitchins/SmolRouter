"""
Access control implementations for model filtering and permissions.

This module provides different access control strategies to control what models
clients can see and access based on IP address, authentication, or other criteria.
"""

import re
import logging
from typing import List, Dict, Any, Optional

from .interfaces import IAccessControl, ModelInfo, ClientContext

logger = logging.getLogger(__name__)


class NoAccessControl(IAccessControl):
    """No-op access control that allows everything"""
    
    async def filter_models(self, models: List[ModelInfo], client: ClientContext) -> List[ModelInfo]:
        """Allow all models"""
        return models
    
    async def can_access_model(self, model: ModelInfo, client: ClientContext) -> bool:
        """Allow access to all models"""
        return True


class IPBasedAccessControl(IAccessControl):
    """Access control based on client IP addresses with pattern matching"""
    
    def __init__(self, rules: Dict[str, Any] = None):
        self.rules = rules or {}
        self._compile_rules()
    
    def _compile_rules(self):
        """Compile IP patterns and model patterns for efficiency"""
        self._compiled_rules = []
        
        for ip_pattern, rule_config in self.rules.items():
            # Handle both simple format and complex format
            if isinstance(rule_config, list):
                # Simple format: ip -> [model_patterns]
                model_patterns = rule_config
                rule = {
                    'ip_pattern': ip_pattern,
                    'allowed_models': model_patterns,
                    'denied_models': [],
                    'allow_all': False
                }
            elif isinstance(rule_config, dict):
                # Complex format with allow/deny lists
                rule = {
                    'ip_pattern': ip_pattern,
                    'allowed_models': rule_config.get('allow', []),
                    'denied_models': rule_config.get('deny', []),
                    'allow_all': rule_config.get('allow_all', False)
                }
            else:
                logger.warning(f"Invalid rule format for IP {ip_pattern}: {rule_config}")
                continue
            
            # Compile regex patterns
            if ip_pattern.startswith('/') and ip_pattern.endswith('/'):
                rule['ip_regex'] = re.compile(ip_pattern[1:-1])
            else:
                rule['ip_regex'] = None
            
            rule['allowed_model_regexes'] = [
                re.compile(p[1:-1]) if p.startswith('/') and p.endswith('/') else None
                for p in rule['allowed_models']
            ]
            
            rule['denied_model_regexes'] = [
                re.compile(p[1:-1]) if p.startswith('/') and p.endswith('/') else None
                for p in rule['denied_models']
            ]
            
            self._compiled_rules.append(rule)
        
        logger.info(f"Compiled {len(self._compiled_rules)} IP-based access control rules")
    
    def _matches_ip(self, client_ip: str, rule: Dict[str, Any]) -> bool:
        """Check if client IP matches rule pattern"""
        if rule['ip_regex']:
            return bool(rule['ip_regex'].match(client_ip))
        else:
            return client_ip == rule['ip_pattern']
    
    def _matches_model_pattern(self, model_id: str, pattern: str, regex_obj: Optional[re.Pattern]) -> bool:
        """Check if model matches pattern"""
        if regex_obj:
            return bool(regex_obj.match(model_id))
        else:
            # Use fnmatch for wildcard patterns (*, ?, [])
            import fnmatch
            return fnmatch.fnmatch(model_id, pattern)
    
    def _find_matching_rule(self, client: ClientContext) -> Optional[Dict[str, Any]]:
        """Find the first matching rule for a client"""
        for rule in self._compiled_rules:
            if self._matches_ip(client.ip, rule):
                return rule
        return None
    
    async def filter_models(self, models: List[ModelInfo], client: ClientContext) -> List[ModelInfo]:
        """Filter models based on IP-based rules"""
        rule = self._find_matching_rule(client)
        if not rule:
            # No specific rule - allow all models
            logger.debug(f"No access rule found for IP {client.ip}, allowing all models")
            return models
        
        logger.debug(f"Applying access rule for IP {client.ip}")
        
        if rule['allow_all']:
            # Allow all models (but still check deny list)
            filtered_models = []
            for model in models:
                if not self._is_model_denied(model, rule):
                    filtered_models.append(model)
            
            logger.debug(f"Allowed {len(filtered_models)}/{len(models)} models for IP {client.ip} (allow_all with deny list)")
            return filtered_models
        
        # Apply allow list filtering
        filtered_models = []
        for model in models:
            if self._is_model_allowed(model, rule) and not self._is_model_denied(model, rule):
                filtered_models.append(model)
        
        logger.debug(f"Allowed {len(filtered_models)}/{len(models)} models for IP {client.ip}")
        return filtered_models
    
    def _is_model_allowed(self, model: ModelInfo, rule: Dict[str, Any]) -> bool:
        """Check if model is in allow list"""
        if not rule['allowed_models']:
            return True  # Empty allow list means allow all
        
        # Check against all possible model identifiers
        model_identifiers = [model.id, model.name, model.display_name] + model.aliases
        
        for i, pattern in enumerate(rule['allowed_models']):
            regex_obj = rule['allowed_model_regexes'][i]
            
            for identifier in model_identifiers:
                if self._matches_model_pattern(identifier, pattern, regex_obj):
                    return True
        
        return False
    
    def _is_model_denied(self, model: ModelInfo, rule: Dict[str, Any]) -> bool:
        """Check if model is in deny list"""
        if not rule['denied_models']:
            return False  # Empty deny list means deny nothing
        
        # Check against all possible model identifiers
        model_identifiers = [model.id, model.name, model.display_name] + model.aliases
        
        for i, pattern in enumerate(rule['denied_models']):
            regex_obj = rule['denied_model_regexes'][i]
            
            for identifier in model_identifiers:
                if self._matches_model_pattern(identifier, pattern, regex_obj):
                    return True
        
        return False
    
    async def can_access_model(self, model: ModelInfo, client: ClientContext) -> bool:
        """Check if client can access specific model"""
        filtered = await self.filter_models([model], client)
        return len(filtered) > 0


class AuthBasedAccessControl(IAccessControl):
    """Access control based on authentication payload"""
    
    def __init__(self, rules: Dict[str, Any] = None):
        self.rules = rules or {}
        self.default_rules = self.rules.get('default', {})
        self.user_rules = self.rules.get('users', {})
        self.role_rules = self.rules.get('roles', {})
    
    def _get_user_id(self, client: ClientContext) -> Optional[str]:
        """Extract user ID from client context"""
        return client.user_id
    
    def _get_user_roles(self, client: ClientContext) -> List[str]:
        """Extract user roles from authentication payload"""
        if not client.auth_payload:
            return []
        
        roles = client.auth_payload.get('roles', [])
        if isinstance(roles, str):
            roles = [roles]
        
        # Also check common role claims
        if 'role' in client.auth_payload:
            role = client.auth_payload['role']
            if isinstance(role, str):
                roles.append(role)
            elif isinstance(role, list):
                roles.extend(role)
        
        return roles
    
    def _find_applicable_rules(self, client: ClientContext) -> Dict[str, Any]:
        """Find applicable access rules for client"""
        # Start with default rules
        applicable_rules = self.default_rules.copy()
        
        # Apply role-based rules
        user_roles = self._get_user_roles(client)
        for role in user_roles:
            if role in self.role_rules:
                role_rules = self.role_rules[role]
                applicable_rules.update(role_rules)
        
        # Apply user-specific rules (highest priority)
        user_id = self._get_user_id(client)
        if user_id and user_id in self.user_rules:
            user_rules = self.user_rules[user_id]
            applicable_rules.update(user_rules)
        
        return applicable_rules
    
    async def filter_models(self, models: List[ModelInfo], client: ClientContext) -> List[ModelInfo]:
        """Filter models based on authentication rules"""
        if not client.auth_payload:
            # No authentication - apply default rules
            rules = self.default_rules
        else:
            rules = self._find_applicable_rules(client)
        
        if not rules:
            # No specific rules - allow all
            return models
        
        allowed_patterns = rules.get('allowed_models', [])
        denied_patterns = rules.get('denied_models', [])
        allow_all = rules.get('allow_all', False)
        
        if allow_all and not denied_patterns:
            return models
        
        filtered_models = []
        for model in models:
            if self._is_model_allowed_by_patterns(model, allowed_patterns, allow_all):
                if not self._is_model_denied_by_patterns(model, denied_patterns):
                    filtered_models.append(model)
        
        user_id = self._get_user_id(client) or "anonymous"
        logger.debug(f"Allowed {len(filtered_models)}/{len(models)} models for user {user_id}")
        return filtered_models
    
    def _is_model_allowed_by_patterns(self, model: ModelInfo, patterns: List[str], allow_all: bool) -> bool:
        """Check if model is allowed by patterns"""
        if allow_all:
            return True
        
        if not patterns:
            return False
        
        model_identifiers = [model.id, model.name, model.display_name] + model.aliases
        
        for pattern in patterns:
            if pattern.startswith('/') and pattern.endswith('/'):
                # Regex pattern
                regex = re.compile(pattern[1:-1])
                for identifier in model_identifiers:
                    if regex.match(identifier):
                        return True
            else:
                # Wildcard match
                import fnmatch
                for identifier in model_identifiers:
                    if fnmatch.fnmatch(identifier, pattern):
                        return True
        
        return False
    
    def _is_model_denied_by_patterns(self, model: ModelInfo, patterns: List[str]) -> bool:
        """Check if model is denied by patterns"""
        if not patterns:
            return False
        
        model_identifiers = [model.id, model.name, model.display_name] + model.aliases
        
        for pattern in patterns:
            if pattern.startswith('/') and pattern.endswith('/'):
                # Regex pattern
                regex = re.compile(pattern[1:-1])
                for identifier in model_identifiers:
                    if regex.match(identifier):
                        return True
            else:
                # Wildcard match
                import fnmatch
                for identifier in model_identifiers:
                    if fnmatch.fnmatch(identifier, pattern):
                        return True
        
        return False
    
    async def can_access_model(self, model: ModelInfo, client: ClientContext) -> bool:
        """Check if client can access specific model"""
        filtered = await self.filter_models([model], client)
        return len(filtered) > 0


class CompositeAccessControl(IAccessControl):
    """Composite access control that combines multiple access control strategies"""
    
    def __init__(self, controls: List[IAccessControl], mode: str = "all"):
        """
        Create composite access control.
        
        Args:
            controls: List of access control implementations
            mode: "all" (all must allow) or "any" (any can allow)
        """
        self.controls = controls
        self.mode = mode
        
        if mode not in ("all", "any"):
            raise ValueError("Mode must be 'all' or 'any'")
    
    async def filter_models(self, models: List[ModelInfo], client: ClientContext) -> List[ModelInfo]:
        """Apply composite filtering"""
        if not self.controls:
            return models
        
        if self.mode == "all":
            # All controls must allow - apply filters sequentially
            filtered_models = models
            for control in self.controls:
                filtered_models = await control.filter_models(filtered_models, client)
            return filtered_models
        
        else:  # mode == "any"
            # Any control can allow - take union of all allowed models
            all_allowed = set()
            for control in self.controls:
                allowed = await control.filter_models(models, client)
                all_allowed.update(model.id for model in allowed)
            
            return [model for model in models if model.id in all_allowed]
    
    async def can_access_model(self, model: ModelInfo, client: ClientContext) -> bool:
        """Check composite access"""
        if not self.controls:
            return True
        
        if self.mode == "all":
            # All controls must allow
            for control in self.controls:
                if not await control.can_access_model(model, client):
                    return False
            return True
        
        else:  # mode == "any"
            # Any control can allow
            for control in self.controls:
                if await control.can_access_model(model, client):
                    return True
            return False


class AccessControlFactory:
    """Factory for creating access control instances"""
    
    @classmethod
    def create_access_control(cls, config: Dict[str, Any] = None) -> IAccessControl:
        """Create access control instance from configuration"""
        if not config:
            return NoAccessControl()
        
        access_type = config.get('type', 'none')
        
        if access_type == 'none':
            return NoAccessControl()
        
        elif access_type == 'ip':
            rules = config.get('rules', {})
            return IPBasedAccessControl(rules)
        
        elif access_type == 'auth':
            rules = config.get('rules', {})
            return AuthBasedAccessControl(rules)
        
        elif access_type == 'composite':
            controls = []
            for control_config in config.get('controls', []):
                control = cls.create_access_control(control_config)
                controls.append(control)
            
            mode = config.get('mode', 'all')
            return CompositeAccessControl(controls, mode)
        
        else:
            raise ValueError(f"Unknown access control type: {access_type}")