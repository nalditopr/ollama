# Gitea Actions Runner for CUDA builds
# Deployed on k3s-arc2 (192.168.98.111)
# Requires Docker socket access for docker build

resource "kubernetes_deployment" "gitea_runner" {
  metadata {
    name      = "gitea-runner"
    namespace = "devops"
    labels = {
      app = "gitea-runner"
    }
  }

  spec {
    replicas = 1
    selector {
      match_labels = {
        app = "gitea-runner"
      }
    }

    template {
      metadata {
        labels = {
          app = "gitea-runner"
        }
      }

      spec {
        # Pin to arc2 node
        node_selector = {
          "kubernetes.io/hostname" = "k3s-arc2"
        }

        containers {
          name  = "runner"
          image = "gitea/act_runner:latest"

          env {
            name  = "GITEA_INSTANCE_URL"
            value = "http://gitea.devops.svc.cluster.local:3000"
          }
          env {
            name  = "GITEA_RUNNER_REGISTRATION_TOKEN"
            value = var.gitea_runner_token
          }
          env {
            name  = "GITEA_RUNNER_NAME"
            value = "cuda-builder"
          }
          env {
            name  = "GITEA_RUNNER_LABELS"
            value = "ubuntu-latest:docker://catthehacker/ubuntu:act-latest"
          }

          volume_mounts {
            name       = "docker-socket"
            mount_path = "/var/run/docker.sock"
          }

          resources {
            requests = {
              cpu    = "4"
              memory = "8Gi"
            }
            limits = {
              cpu    = "20"
              memory = "28Gi"
            }
          }
        }

        volumes {
          name = "docker-socket"
          host_path {
            path = "/var/run/docker.sock"
            type = "Socket"
          }
        }
      }
    }
  }
}

variable "gitea_runner_token" {
  description = "Gitea Actions runner registration token"
  type        = string
  sensitive   = true
}
