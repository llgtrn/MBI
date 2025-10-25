# Secrets Management Infrastructure
# GCP Secret Manager with auto-rotation policies

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod"
  }
}

variable "rotation_interval_days" {
  description = "Secret rotation interval in days"
  type        = number
  default     = 90
  validation {
    condition     = var.rotation_interval_days >= 30
    error_message = "Rotation interval must be at least 30 days"
  }
}

# Enable Secret Manager API
resource "google_project_service" "secretmanager" {
  project = var.project_id
  service = "secretmanager.googleapis.com"

  disable_on_destroy = false
}

# Secret definitions with rotation policies
locals {
  secrets = {
    meta-api-key = {
      description = "Meta Marketing API Key"
      rotation    = var.rotation_interval_days
    }
    google-ads-api-key = {
      description = "Google Ads API Key"
      rotation    = var.rotation_interval_days
    }
    tiktok-api-key = {
      description = "TikTok Ads API Key"
      rotation    = var.rotation_interval_days
    }
    shopify-api-key = {
      description = "Shopify API Key"
      rotation    = var.rotation_interval_days
    }
    anthropic-api-key = {
      description = "Anthropic Claude API Key"
      rotation    = var.rotation_interval_days
    }
    openai-api-key = {
      description = "OpenAI API Key"
      rotation    = var.rotation_interval_days
    }
    bigquery-sa-key = {
      description = "BigQuery Service Account Key"
      rotation    = var.rotation_interval_days
    }
    redis-password = {
      description = "Redis Password"
      rotation    = 30  # More frequent for infra secrets
    }
    postgres-password = {
      description = "PostgreSQL Password"
      rotation    = 30
    }
    hmac-signing-key = {
      description = "HMAC Signing Key for Webhooks"
      rotation    = 30
    }
  }
}

# Create secrets
resource "google_secret_manager_secret" "secrets" {
  for_each = local.secrets

  project   = var.project_id
  secret_id = "${var.environment}-${each.key}"

  replication {
    auto {}
  }

  rotation {
    next_rotation_time = timeadd(timestamp(), "${each.value.rotation * 24}h")
    rotation_period    = "${each.value.rotation * 24}h"
  }

  labels = {
    environment       = var.environment
    managed_by        = "terraform"
    rotation_days     = tostring(each.value.rotation)
    auto_rotation     = "enabled"
  }

  depends_on = [google_project_service.secretmanager]
}

# IAM bindings for service accounts
resource "google_secret_manager_secret_iam_member" "mbi_workload_access" {
  for_each = google_secret_manager_secret.secrets

  project   = var.project_id
  secret_id = each.value.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:mbi-workload-${var.environment}@${var.project_id}.iam.gserviceaccount.com"
}

# Cloud Function for rotation notifications
resource "google_cloudfunctions_function" "rotation_notifier" {
  name        = "secret-rotation-notifier-${var.environment}"
  description = "Notifies team when secrets need rotation"
  runtime     = "python39"
  project     = var.project_id

  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.rotation_functions.name
  source_archive_object = google_storage_bucket_object.rotation_function_code.name
  trigger_http          = true
  entry_point           = "notify_rotation"

  environment_variables = {
    ENVIRONMENT   = var.environment
    SLACK_WEBHOOK = google_secret_manager_secret.secrets["slack-webhook"].secret_id
  }

  labels = {
    environment = var.environment
  }
}

# Storage bucket for Cloud Functions
resource "google_storage_bucket" "rotation_functions" {
  name     = "${var.project_id}-rotation-functions-${var.environment}"
  project  = var.project_id
  location = "US"

  uniform_bucket_level_access = true

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 30
    }
  }
}

# Placeholder for function code (deploy separately)
resource "google_storage_bucket_object" "rotation_function_code" {
  name   = "rotation-notifier-${timestamp()}.zip"
  bucket = google_storage_bucket.rotation_functions.name
  source = "${path.module}/functions/rotation_notifier.zip"
}

# Cloud Scheduler for periodic rotation checks
resource "google_cloud_scheduler_job" "rotation_checker" {
  name        = "secret-rotation-checker-${var.environment}"
  description = "Daily check for secrets needing rotation"
  project     = var.project_id
  region      = "us-central1"
  schedule    = "0 9 * * *" # Daily at 9 AM UTC

  http_target {
    http_method = "POST"
    uri         = google_cloudfunctions_function.rotation_notifier.https_trigger_url

    oidc_token {
      service_account_email = "mbi-workload-${var.environment}@${var.project_id}.iam.gserviceaccount.com"
    }
  }
}

# Monitoring for secret access
resource "google_monitoring_alert_policy" "secret_access_anomaly" {
  display_name = "Secret Access Anomaly - ${var.environment}"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "High secret access rate"

    condition_threshold {
      filter          = "resource.type=\"secret_manager_secret\" AND metric.type=\"secretmanager.googleapis.com/secret/access_count\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 100

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.id]
  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_notification_channel" "email" {
  display_name = "MBI Ops Email - ${var.environment}"
  type         = "email"
  project      = var.project_id

  labels = {
    email_address = "mbi-ops@company.com"
  }
}

# Outputs
output "secret_ids" {
  description = "Map of secret names to their IDs"
  value = {
    for k, v in google_secret_manager_secret.secrets : k => v.secret_id
  }
}

output "rotation_function_url" {
  description = "URL of rotation notifier function"
  value       = google_cloudfunctions_function.rotation_notifier.https_trigger_url
  sensitive   = true
}

output "secret_access_instructions" {
  description = "Instructions for accessing secrets"
  value = <<-EOT
    Secrets are stored in GCP Secret Manager with auto-rotation enabled.
    
    To access a secret from code:
      from src.config.secrets import get_secret
      api_key = get_secret("META_API_KEY")
    
    To manually retrieve a secret:
      gcloud secrets versions access latest --secret="${var.environment}-meta-api-key"
    
    Rotation schedule: Every ${var.rotation_interval_days} days
    Notifications: Sent to mbi-ops@company.com when rotation is due
  EOT
}
