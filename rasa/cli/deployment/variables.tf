variable "project" {
  type        = string
  description = "The id of google cloud project."
}

variable "zones" {
  type        = list(string)
  description = "The zones of google cloud project."
  default = ["europe-west1-c"]
}

variable "region" {
  type        = string
  description = "The region of the google cloud project."
  default = "europe-west1"
}

variable "name" {
  type        = string
  description = "The name of the cluster."
  default = "rasa"
}
