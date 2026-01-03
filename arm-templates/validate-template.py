#!/usr/bin/env python3
"""
Validate ARM template before deployment.

This script validates the ARM template JSON syntax and checks for common issues
that might prevent successful deployment to Azure.

Usage:
    python arm-templates/validate-template.py
"""

import json
import sys
from pathlib import Path


def validate_json_syntax(template_path):
    """Validate that the file is valid JSON."""
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for BOM
        if content.startswith('\ufeff'):
            print("⚠️  Warning: File contains BOM (Byte Order Mark)")
            content = content.lstrip('\ufeff')
            
        # Parse JSON
        template = json.loads(content)
        print("✅ Valid JSON syntax")
        return template, content
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return None, None
    except FileNotFoundError:
        print(f"❌ File not found: {template_path}")
        return None, None


def validate_arm_structure(template):
    """Validate ARM template structure."""
    issues = []
    
    # Check required fields
    required_fields = ['$schema', 'contentVersion', 'resources']
    for field in required_fields:
        if field not in template:
            issues.append(f"Missing required field: {field}")
    
    # Check schema
    schema = template.get('$schema', '')
    if 'deploymentTemplate.json' not in schema:
        issues.append(f"Invalid schema: {schema}")
    
    # Check content version
    content_version = template.get('contentVersion', '')
    if not content_version:
        issues.append("Missing contentVersion")
    
    # Check resources
    resources = template.get('resources', [])
    if not isinstance(resources, list):
        issues.append("Resources must be an array")
    elif len(resources) == 0:
        issues.append("No resources defined")
    
    return issues


def check_api_versions(template):
    """Check resource API versions."""
    resources = template.get('resources', [])
    api_info = {}
    
    for resource in resources:
        resource_type = resource.get('type', 'Unknown')
        api_version = resource.get('apiVersion', 'Missing')
        
        if resource_type not in api_info:
            api_info[resource_type] = []
        if api_version not in api_info[resource_type]:
            api_info[resource_type].append(api_version)
    
    return api_info


def validate_resource_dependencies(template):
    """Check for dependency issues."""
    resources = template.get('resources', [])
    resource_names = set()
    issues = []
    
    # Collect resource names
    for resource in resources:
        name = resource.get('name', '')
        if isinstance(name, str) and not name.startswith('['):
            resource_names.add(name)
    
    # Check dependencies
    for resource in resources:
        depends_on = resource.get('dependsOn', [])
        for dependency in depends_on:
            # Skip ARM expressions
            if not isinstance(dependency, str) or dependency.startswith('['):
                continue
            # Simplified check - real validation would parse ARM expressions
    
    return issues


def main():
    """Main validation function."""
    template_path = Path(__file__).parent / 'azuredeploy.json'
    
    print("=" * 60)
    print("ARM Template Validation")
    print("=" * 60)
    print(f"Template: {template_path}")
    print()
    
    # Validate JSON syntax
    template, content = validate_json_syntax(template_path)
    if template is None:
        sys.exit(1)
    
    # Validate ARM structure
    print("\nValidating ARM template structure...")
    structural_issues = validate_arm_structure(template)
    if structural_issues:
        for issue in structural_issues:
            print(f"❌ {issue}")
        sys.exit(1)
    else:
        print("✅ ARM template structure is valid")
    
    # Check API versions
    print("\nResource API versions:")
    api_info = check_api_versions(template)
    for resource_type, versions in sorted(api_info.items()):
        for version in versions:
            print(f"  • {resource_type}: {version}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Schema: {template.get('$schema', 'N/A')}")
    print(f"Content Version: {template.get('contentVersion', 'N/A')}")
    print(f"Parameters: {len(template.get('parameters', {}))}")
    print(f"Resources: {len(template.get('resources', []))}")
    print(f"Outputs: {len(template.get('outputs', {}))}")
    print(f"File Size: {len(content):,} bytes")
    
    print("\n✅ Template validation passed!")
    print("\n💡 Tip: You can now deploy this template using:")
    print("   • ./arm-templates/deploy.sh (Linux/Mac)")
    print("   • .\\arm-templates\\deploy.bat (Windows)")
    print("   • Azure Portal custom deployment (if button fails)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
