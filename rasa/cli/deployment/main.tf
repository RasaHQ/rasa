terraform {
  backend "gcs" {
    bucket      = "${PROJECT_NAME}_${PACKAGE_NAME_DOCKER_TERRAFORRM}_state"
    prefix      = "env/dev"
  }
}

locals {
  project_id             = var.project
  cluster_name           = var.name
  region                 = var.region
  zones                  = var.zones
  ip_range_pods_name     = "${local.cluster_name}-subnet-pods"
  ip_range_services_name = "${local.cluster_name}-subnet-services"
}

# Setup VPC and networking
module "vpc" {
  source       = "terraform-google-modules/network/google"
  version      = "~> 2.4"
  project_id   = local.project_id
  network_name = local.cluster_name

  subnets = [
    {
      subnet_name   = "${local.cluster_name}-subnet"
      subnet_ip     = "10.35.0.0/16"
      subnet_region = local.region
    },
  ]

  secondary_ranges = {
    "${local.cluster_name}-subnet" = [
      {
        range_name    = local.ip_range_pods_name
        ip_cidr_range = "192.168.0.0/18"
      },
      {
        range_name    = local.ip_range_services_name
        ip_cidr_range = "192.168.64.0/18"
      },
    ]
  }
}

module "gke" {
  source                     = "terraform-google-modules/kubernetes-engine/google"
  project_id                 = local.project_id
  region                     = local.region
  name                       = local.cluster_name
  zones                      = local.zones
  network                    = module.vpc.network_name
  subnetwork                 = module.vpc.subnets_names[0]
  ip_range_pods              = local.ip_range_pods_name
  ip_range_services          = local.ip_range_services_name
  horizontal_pod_autoscaling = true
  description                = "Cluster to run rasa"
  grant_registry_access      = true

  node_pools = [
    {
      name               = "default-node-pool"
      machine_type       = "n1-standard-2"
      min_count          = 1
      max_count          = 2
      disk_size_gb       = 20
      disk_type          = "pd-standard"
      image_type         = "UBUNTU"
      auto_repair        = true
      auto_upgrade       = true
      autoscaling        = true
      preemptible        = false
      initial_node_count = 1
    },
  ]

  node_pools_oauth_scopes = {
    all = []

    default-node-pool = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}
