reset_system_service() {

  # Restart apache2
  sudo systemctl stop apache2
  sudo systemctl start apache2
  echo "Apache2 restarted successfully."


  echo "Now working on $1 service..."
  # Restart service
  local service_name="$1.service"

  if [[ -z "$1" ]]; then
    echo "Error: No service name provided. Usage: reset_system_service <service_name>"
    return 1
  fi

  echo "Reloading system daemon..."
  sudo systemctl daemon-reload

  echo "Enabling $service_name..."
  sudo systemctl enable "$service_name"
  if [[ $? -ne 0 ]]; then
    echo "Failed to enable $service_name"
    return 1
  fi

  echo "Starting $service_name..."
  sudo systemctl start "$service_name"
  if [[ $? -ne 0 ]]; then
    echo "Failed to start $service_name"
    sudo systemctl status "$service_name" --no-pager
    return 1
  fi

  echo "$service_name started successfully."
}

reset_system_service "queue"