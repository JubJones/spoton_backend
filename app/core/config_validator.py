"""
Configuration validation and documentation system.

Comprehensive validation framework for Phase 6 consolidated configuration
with detailed validation rules, documentation generation, and health checks.
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from enum import Enum
from datetime import datetime
import logging
import json
import os
from dataclasses import dataclass

from app.core.config_consolidated import ConsolidatedSettings, validate_environment_config

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation result severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Individual validation result."""
    rule_name: str
    severity: ValidationSeverity
    message: str
    component: str
    suggestion: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'rule_name': self.rule_name,
            'severity': self.severity.value,
            'message': self.message,
            'component': self.component,
            'suggestion': self.suggestion,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class ConfigurationValidator:
    """
    Comprehensive configuration validation system.
    
    Validates all aspects of the consolidated configuration system
    with detailed rules and recommendations.
    """
    
    def __init__(self, settings: ConsolidatedSettings):
        """Initialize validator with settings instance."""
        self.settings = settings
        self.validation_rules: List[Callable] = []
        self._register_validation_rules()
    
    def _register_validation_rules(self) -> None:
        """Register all validation rules."""
        self.validation_rules = [
            self._validate_core_configuration,
            self._validate_ai_model_configuration,
            self._validate_database_configuration,
            self._validate_storage_configuration,
            self._validate_camera_configuration,
            self._validate_cross_component_consistency,
            self._validate_security_settings,
            self._validate_performance_settings,
            self._validate_environment_specific_rules
        ]
    
    def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation rules and return comprehensive results.
        
        Returns:
            Dictionary containing validation summary and detailed results
        """
        all_results = []
        validation_summary = {
            'total_rules': len(self.validation_rules),
            'passed': 0,
            'warnings': 0,
            'errors': 0,
            'critical': 0,
            'validation_timestamp': datetime.utcnow().isoformat()
        }
        
        # Run each validation rule
        for rule in self.validation_rules:
            try:
                rule_results = rule()
                all_results.extend(rule_results)
                
                # Update summary counts
                for result in rule_results:
                    if result.severity == ValidationSeverity.INFO:
                        validation_summary['passed'] += 1
                    elif result.severity == ValidationSeverity.WARNING:
                        validation_summary['warnings'] += 1
                    elif result.severity == ValidationSeverity.ERROR:
                        validation_summary['errors'] += 1
                    elif result.severity == ValidationSeverity.CRITICAL:
                        validation_summary['critical'] += 1
                
            except Exception as e:
                error_result = ValidationResult(
                    rule_name=rule.__name__,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation rule failed: {e}",
                    component="validator",
                    suggestion="Check validator implementation"
                )
                all_results.append(error_result)
                validation_summary['errors'] += 1
        
        # Calculate overall validation status
        validation_summary['overall_status'] = self._calculate_overall_status(validation_summary)
        
        return {
            'summary': validation_summary,
            'results': [result.to_dict() for result in all_results],
            'recommendations': self._generate_recommendations(all_results),
            'environment_validation': validate_environment_config()
        }
    
    def _calculate_overall_status(self, summary: Dict[str, Any]) -> str:
        """Calculate overall validation status."""
        if summary['critical'] > 0:
            return "critical"
        elif summary['errors'] > 0:
            return "failed"
        elif summary['warnings'] > 0:
            return "warning"
        else:
            return "passed"
    
    def _validate_core_configuration(self) -> List[ValidationResult]:
        """Validate core application configuration."""
        results = []
        core = self.settings.core
        
        # Environment validation
        if core.environment.value == "development" and not core.debug:
            results.append(ValidationResult(
                rule_name="core_dev_debug_mode",
                severity=ValidationSeverity.WARNING,
                message="Debug mode disabled in development environment",
                component="core",
                suggestion="Consider enabling debug mode for development"
            ))
        
        # Security validation
        if core.environment.value == "production":
            if core.secret_key == "dev-secret-key-change-in-production":
                results.append(ValidationResult(
                    rule_name="core_production_secret_key",
                    severity=ValidationSeverity.CRITICAL,
                    message="Default secret key used in production",
                    component="core",
                    suggestion="Set a secure SECRET_KEY environment variable"
                ))
            
            if core.debug:
                results.append(ValidationResult(
                    rule_name="core_production_debug_mode",
                    severity=ValidationSeverity.ERROR,
                    message="Debug mode enabled in production",
                    component="core",
                    suggestion="Set DEBUG=false in production environment"
                ))
        
        # Resource limits validation
        if core.max_concurrent_tasks < 5:
            results.append(ValidationResult(
                rule_name="core_concurrent_tasks_low",
                severity=ValidationSeverity.WARNING,
                message=f"Low concurrent task limit: {core.max_concurrent_tasks}",
                component="core",
                suggestion="Consider increasing MAX_CONCURRENT_TASKS for better performance"
            ))
        
        # API configuration validation
        if "*" in core.cors_origins and core.environment.value == "production":
            results.append(ValidationResult(
                rule_name="core_cors_wildcard_production",
                severity=ValidationSeverity.ERROR,
                message="Wildcard CORS origins in production",
                component="core",
                suggestion="Restrict CORS_ORIGINS to specific domains"
            ))
        
        return results
    
    def _validate_ai_model_configuration(self) -> List[ValidationResult]:
        """Validate AI model configuration."""
        results = []
        
        try:
            ai_config = self.settings.ai_config
            
            # Performance validation
            if ai_config.target_fps > 30 and not ai_config.detector.use_amp:
                results.append(ValidationResult(
                    rule_name="ai_high_fps_no_amp",
                    severity=ValidationSeverity.WARNING,
                    message="High FPS target without AMP may impact performance",
                    component="ai_models",
                    suggestion="Consider enabling AMP for better performance"
                ))
            
            # Model weight validation
            weight_validation = ai_config.validate_model_weights_exist()
            for component, exists in weight_validation.items():
                if not exists and 'weights' in component:
                    results.append(ValidationResult(
                        rule_name=f"ai_weights_{component}",
                        severity=ValidationSeverity.ERROR,
                        message=f"AI model weights not found: {component}",
                        component="ai_models",
                        suggestion="Download required model weights or check paths"
                    ))
            
            # Configuration consistency validation
            if ai_config.reid.similarity_threshold > 0.9:
                results.append(ValidationResult(
                    rule_name="ai_reid_threshold_high",
                    severity=ValidationSeverity.WARNING,
                    message="Very high ReID similarity threshold may miss valid matches",
                    component="ai_models",
                    suggestion="Consider lowering REID_SIMILARITY_THRESHOLD"
                ))
            
            if ai_config.detector.confidence_threshold < 0.3:
                results.append(ValidationResult(
                    rule_name="ai_detector_threshold_low",
                    severity=ValidationSeverity.WARNING,
                    message="Low detection confidence threshold may increase false positives",
                    component="ai_models",
                    suggestion="Consider increasing DETECTION_CONFIDENCE_THRESHOLD"
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                rule_name="ai_config_access",
                severity=ValidationSeverity.ERROR,
                message=f"Cannot access AI configuration: {e}",
                component="ai_models",
                suggestion="Check AI model configuration module"
            ))
        
        return results
    
    def _validate_database_configuration(self) -> List[ValidationResult]:
        """Validate database configuration."""
        results = []
        
        try:
            db_config = self.settings.database_config
            
            # Connection pool validation
            total_connections = (db_config.postgresql.pool_size + 
                               db_config.postgresql.max_overflow + 
                               db_config.redis.connection_pool_size)
            
            if total_connections > 200:
                results.append(ValidationResult(
                    rule_name="db_high_connection_count",
                    severity=ValidationSeverity.WARNING,
                    message=f"High total database connections: {total_connections}",
                    component="database",
                    suggestion="Monitor database connection usage and adjust pool sizes"
                ))
            
            # SSL validation for production
            if (self.settings.core.environment.value == "production" and 
                db_config.postgresql.ssl_mode == "disable"):
                results.append(ValidationResult(
                    rule_name="db_ssl_disabled_production",
                    severity=ValidationSeverity.ERROR,
                    message="SSL disabled for PostgreSQL in production",
                    component="database",
                    suggestion="Enable SSL with ssl_mode='require' or 'verify-full'"
                ))
            
            # Timeout validation
            if db_config.postgresql.statement_timeout < 30:
                results.append(ValidationResult(
                    rule_name="db_low_statement_timeout",
                    severity=ValidationSeverity.WARNING,
                    message="Very low SQL statement timeout may cause query failures",
                    component="database",
                    suggestion="Increase POSTGRES_STATEMENT_TIMEOUT"
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                rule_name="db_config_access",
                severity=ValidationSeverity.ERROR,
                message=f"Cannot access database configuration: {e}",
                component="database",
                suggestion="Check database configuration module"
            ))
        
        return results
    
    def _validate_storage_configuration(self) -> List[ValidationResult]:
        """Validate storage configuration."""
        results = []
        
        try:
            storage_config = self.settings.storage_config
            
            # S3 credentials validation
            if (storage_config.storage_backend.value in ["s3", "hybrid"] and 
                not storage_config.s3.access_key_id):
                results.append(ValidationResult(
                    rule_name="storage_s3_credentials",
                    severity=ValidationSeverity.ERROR,
                    message="S3 backend selected but credentials not configured",
                    component="storage",
                    suggestion="Configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
                ))
            
            # Local storage limits validation
            if storage_config.local.max_storage_size_gb < 10:
                results.append(ValidationResult(
                    rule_name="storage_local_size_low",
                    severity=ValidationSeverity.WARNING,
                    message=f"Low local storage limit: {storage_config.local.max_storage_size_gb}GB",
                    component="storage",
                    suggestion="Consider increasing local storage limit"
                ))
            
            # Export settings validation
            if storage_config.exports.max_export_size_mb > 1000:
                results.append(ValidationResult(
                    rule_name="storage_large_export_size",
                    severity=ValidationSeverity.WARNING,
                    message="Large export size limit may impact performance",
                    component="storage",
                    suggestion="Consider reducing MAX_EXPORT_SIZE_MB"
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                rule_name="storage_config_access",
                severity=ValidationSeverity.ERROR,
                message=f"Cannot access storage configuration: {e}",
                component="storage",
                suggestion="Check storage configuration module"
            ))
        
        return results
    
    def _validate_camera_configuration(self) -> List[ValidationResult]:
        """Validate camera configuration."""
        results = []
        
        try:
            camera_config = self.settings.camera_config
            
            # Homography directory validation
            if not camera_config.resolved_homography_base_path.exists():
                results.append(ValidationResult(
                    rule_name="camera_homography_dir",
                    severity=ValidationSeverity.ERROR,
                    message="Homography data directory not found",
                    component="cameras",
                    suggestion=f"Create directory: {camera_config.homography_data_dir}"
                ))
            
            # Video set validation
            if not camera_config.video_sets:
                results.append(ValidationResult(
                    rule_name="camera_no_video_sets",
                    severity=ValidationSeverity.WARNING,
                    message="No video sets configured",
                    component="cameras",
                    suggestion="Configure VIDEO_SETS for camera processing"
                ))
            
            # Cross-validation of video sets and handoffs
            video_set_count = len(camera_config.video_sets)
            handoff_count = len(camera_config.camera_handoff_details)
            
            if video_set_count != handoff_count:
                results.append(ValidationResult(
                    rule_name="camera_config_mismatch",
                    severity=ValidationSeverity.WARNING,
                    message=f"Video sets ({video_set_count}) and handoff configs ({handoff_count}) count mismatch",
                    component="cameras",
                    suggestion="Ensure each video set has corresponding handoff configuration"
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                rule_name="camera_config_access",
                severity=ValidationSeverity.ERROR,
                message=f"Cannot access camera configuration: {e}",
                component="cameras",
                suggestion="Check camera configuration module"
            ))
        
        return results
    
    def _validate_cross_component_consistency(self) -> List[ValidationResult]:
        """Validate consistency across configuration components."""
        results = []
        
        try:
            # Validate performance settings alignment
            ai_fps = self.settings.ai_config.target_fps
            if ai_fps > 30 and self.settings.core.max_concurrent_tasks < 10:
                results.append(ValidationResult(
                    rule_name="cross_performance_mismatch",
                    severity=ValidationSeverity.WARNING,
                    message="High AI FPS with low concurrent task limit may create bottlenecks",
                    component="cross_component",
                    suggestion="Increase MAX_CONCURRENT_TASKS or reduce AI_TARGET_FPS"
                ))
            
            # Validate storage and processing alignment
            storage_backend = self.settings.storage_config.storage_backend.value
            if storage_backend == "s3" and not self.settings.storage_config.enable_caching:
                results.append(ValidationResult(
                    rule_name="cross_s3_no_cache",
                    severity=ValidationSeverity.WARNING,
                    message="S3 storage without caching may impact performance",
                    component="cross_component",
                    suggestion="Enable storage caching for S3 backend"
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                rule_name="cross_component_validation",
                severity=ValidationSeverity.ERROR,
                message=f"Cross-component validation failed: {e}",
                component="cross_component",
                suggestion="Check configuration component accessibility"
            ))
        
        return results
    
    def _validate_security_settings(self) -> List[ValidationResult]:
        """Validate security-related settings."""
        results = []
        
        # JWT/Token security
        if self.settings.core.access_token_expire_minutes > 1440:  # 24 hours
            results.append(ValidationResult(
                rule_name="security_long_token_expiry",
                severity=ValidationSeverity.WARNING,
                message="Very long access token expiry may pose security risk",
                component="security",
                suggestion="Consider shorter ACCESS_TOKEN_EXPIRE_MINUTES"
            ))
        
        # Environment-specific security checks
        if self.settings.core.environment.value == "production":
            try:
                db_config = self.settings.database_config
                if db_config.postgresql.password == "spoton_password":
                    results.append(ValidationResult(
                        rule_name="security_default_db_password",
                        severity=ValidationSeverity.CRITICAL,
                        message="Default database password in production",
                        component="security",
                        suggestion="Set secure POSTGRES_PASSWORD"
                    ))
            except:
                pass
        
        return results
    
    def _validate_performance_settings(self) -> List[ValidationResult]:
        """Validate performance-related settings."""
        results = []
        
        try:
            # Memory and resource validation
            ai_config = self.settings.ai_config
            db_config = self.settings.database_config
            
            # High performance configuration check
            if (ai_config.target_fps > 25 and 
                db_config.postgresql.pool_size < 15):
                results.append(ValidationResult(
                    rule_name="perf_high_fps_small_pool",
                    severity=ValidationSeverity.WARNING,
                    message="High FPS processing with small database pool",
                    component="performance",
                    suggestion="Increase database pool size for high throughput"
                ))
            
            # AI model performance settings
            if (ai_config.detector.use_amp and 
                ai_config.tracker.half_precision and
                ai_config.reid.half_precision):
                results.append(ValidationResult(
                    rule_name="perf_all_precision_optimized",
                    severity=ValidationSeverity.INFO,
                    message="All AI models configured for performance optimization",
                    component="performance"
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                rule_name="perf_validation_error",
                severity=ValidationSeverity.WARNING,
                message=f"Performance validation incomplete: {e}",
                component="performance"
            ))
        
        return results
    
    def _validate_environment_specific_rules(self) -> List[ValidationResult]:
        """Validate environment-specific configuration rules."""
        results = []
        
        env = self.settings.core.environment.value
        
        if env == "development":
            # Development-specific validations
            if not self.settings.core.debug:
                results.append(ValidationResult(
                    rule_name="env_dev_no_debug",
                    severity=ValidationSeverity.INFO,
                    message="Debug mode disabled in development",
                    component="environment",
                    suggestion="Enable debug mode for development convenience"
                ))
        
        elif env == "production":
            # Production-specific validations
            if self.settings.core.log_level.value == "DEBUG":
                results.append(ValidationResult(
                    rule_name="env_prod_debug_logging",
                    severity=ValidationSeverity.WARNING,
                    message="Debug logging in production may impact performance",
                    component="environment",
                    suggestion="Use INFO or WARNING log level in production"
                ))
        
        return results
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Count issues by severity
        error_count = sum(1 for r in results if r.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for r in results if r.severity == ValidationSeverity.WARNING)
        critical_count = sum(1 for r in results if r.severity == ValidationSeverity.CRITICAL)
        
        # Generate priority recommendations
        if critical_count > 0:
            recommendations.append(f"🚨 URGENT: Address {critical_count} critical security/configuration issues immediately")
        
        if error_count > 0:
            recommendations.append(f"❌ Fix {error_count} configuration errors before deployment")
        
        if warning_count > 0:
            recommendations.append(f"⚠️  Review {warning_count} configuration warnings for optimization")
        
        # Component-specific recommendations
        components_with_issues = set(r.component for r in results if r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
        
        if "core" in components_with_issues:
            recommendations.append("🔧 Review core application settings and security configuration")
        
        if "ai_models" in components_with_issues:
            recommendations.append("🤖 Verify AI model weights and performance settings")
        
        if "database" in components_with_issues:
            recommendations.append("🗄️  Check database connection and performance configuration")
        
        if "storage" in components_with_issues:
            recommendations.append("💾 Validate storage backend and credential configuration")
        
        # Environment-specific recommendations
        env = self.settings.core.environment.value
        if env == "production":
            recommendations.extend([
                "🔒 Ensure all production security requirements are met",
                "📊 Configure monitoring and alerting for production environment",
                "🔄 Test configuration in staging environment before deployment"
            ])
        
        return recommendations


def generate_configuration_documentation(settings: ConsolidatedSettings) -> Dict[str, Any]:
    """
    Generate comprehensive configuration documentation.
    
    Args:
        settings: Consolidated settings instance
        
    Returns:
        Dictionary containing complete configuration documentation
    """
    documentation = {
        'metadata': {
            'generated_at': datetime.utcnow().isoformat(),
            'environment': settings.core.environment.value,
            'version': '1.0.0'
        },
        'overview': {
            'description': 'SpotOn Backend Configuration Documentation',
            'architecture': 'Phase 6 Consolidated Configuration System',
            'components': ['core', 'ai_models', 'database', 'storage', 'cameras']
        },
        'environments': {
            'development': {
                'purpose': 'Local development with debug features',
                'characteristics': ['Debug enabled', 'Local storage', 'Relaxed security'],
                'recommended_for': ['Development', 'Testing', 'Debugging']
            },
            'testing': {
                'purpose': 'Automated testing environment',
                'characteristics': ['Isolated resources', 'Fast execution', 'Test databases'],
                'recommended_for': ['CI/CD', 'Unit tests', 'Integration tests']
            },
            'staging': {
                'purpose': 'Pre-production validation',
                'characteristics': ['Production-like', 'Enhanced monitoring', 'Safe testing'],
                'recommended_for': ['UAT', 'Performance testing', 'Deployment validation']
            },
            'production': {
                'purpose': 'Live production deployment',
                'characteristics': ['Security hardened', 'Performance optimized', 'Monitoring enabled'],
                'recommended_for': ['Production workloads', 'Customer-facing services']
            }
        },
        'configuration_summary': settings.get_configuration_summary(),
        'validation_results': ConfigurationValidator(settings).validate_all()
    }
    
    return documentation


def export_configuration_documentation(
    settings: ConsolidatedSettings, 
    output_path: Optional[str] = None
) -> str:
    """
    Export configuration documentation to file.
    
    Args:
        settings: Consolidated settings instance
        output_path: Optional output file path
        
    Returns:
        Path to exported documentation file
    """
    if output_path is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = f"./config_documentation_{timestamp}.json"
    
    documentation = generate_configuration_documentation(settings)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documentation, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Configuration documentation exported to: {output_file}")
    return str(output_file)


def validate_configuration_health() -> Dict[str, Any]:
    """
    Perform health check of current configuration.
    
    Returns:
        Dictionary containing health check results
    """
    from app.core.config_consolidated import get_settings
    
    try:
        settings = get_settings()
        validator = ConfigurationValidator(settings)
        validation_results = validator.validate_all()
        
        # Determine overall health status
        if validation_results['summary']['critical'] > 0:
            health_status = "unhealthy"
        elif validation_results['summary']['errors'] > 0:
            health_status = "degraded"
        elif validation_results['summary']['warnings'] > 5:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            'status': health_status,
            'timestamp': datetime.utcnow().isoformat(),
            'environment': settings.core.environment.value,
            'summary': validation_results['summary'],
            'critical_issues': [
                r for r in validation_results['results'] 
                if r['severity'] == 'critical'
            ],
            'recommendations': validation_results['recommendations'][:5]  # Top 5
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e),
            'message': 'Configuration health check failed'
        }