stage('Data Prep') {
  steps {
    bat 'docker build -t dataprep-app -f Dockerfile.prep .'
    bat 'docker run --rm dataprep-app'
  }
}
