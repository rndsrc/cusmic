# One build matrix controls toolkit/CuPy pins, GPU code targets and image tags.
variable "VERSION" { default = "local" }
variable "REVISION" { default = "dev" }
variable "PLATFORM" { default = "linux/arm64/v8" }
variable "CUDA_ARCHS" { default = "" }
variable "CUPY_VERSION" { default = "" }

# Add linux/amd64 through PLATFORM after qualifying that build on a GPU host.
target "images" {
  name = "${variant.name}-${role.key}"
  matrix = {
    variant = [
      {
        name       = "cuda12"
        major      = "12"
        toolkit    = "12.6.3"
	cupy_version = "14.2.0"
        cuda_archs = "87"
        suffix     = "-cuda12"
      },
      {
        name       = "cuda13"
        major      = "13"
        toolkit    = "13.0.2"
	cupy_version = "14.2.0"
        cuda_archs = "87 121"
        suffix     = ""
      },
    ]
    role = [
      { key = "cupysmic",      stage = "cupy-cli", image = "cupysmic", suffix = "" },
      { key = "cupysmic-slim", stage = "cupy-api", image = "cupysmic", suffix = "-slim" },
      { key = "cudasmic",      stage = "cuda-cli", image = "cudasmic", suffix = "" },
      { key = "cudasmic-slim", stage = "cuda-api", image = "cudasmic", suffix = "-slim" },
    ]
  }

  context    = "."
  dockerfile = "Dockerfile"
  target     = role.stage
  args = {
    VERSION       = VERSION
    REVISION      = REVISION
    CUPY_VERSION  = CUPY_VERSION != "" ? CUPY_VERSION : variant.cupy_version
    CUDA_ARCHS    = CUDA_ARCHS != "" ? CUDA_ARCHS : variant.cuda_archs
    CUDA_MAJOR    = variant.major
    CUDA_TOOLKIT  = variant.toolkit
  }
  platforms = [PLATFORM]
  tags = ["rndsrc/${role.image}:${VERSION}${role.suffix}${variant.suffix}"]
}

group "cuda12" {
  targets = [
    "cuda12-cupysmic", "cuda12-cupysmic-slim",
    "cuda12-cudasmic", "cuda12-cudasmic-slim",
  ]
}

group "cuda13" {
  targets = [
    "cuda13-cupysmic", "cuda13-cupysmic-slim",
    "cuda13-cudasmic", "cuda13-cudasmic-slim",
  ]
}

group "default" {
  targets = [
    "cuda13-cupysmic", "cuda13-cupysmic-slim",
    "cuda13-cudasmic", "cuda13-cudasmic-slim",
  ]
}
