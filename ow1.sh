curl -X 'GET' 'https://api.denkers.co/queue/$1?user_id=1' 
    -H 'user_id: 1' 
    -H 'x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b' 
    -H 'accept: application/json' 
    -H 'Content-Type: application/json'


function ow() {
    local endpoint="$1"
    local id="${2:-1}"  # Default to 1 if no ID provided
    local user_id="${3:-1}"  # Default to 1 if no user_id provided
    local token="6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b"
    
    curl -X 'GET' "https://api.denkers.co/${endpoint}/${id}?user_id=${user_id}" \
        -H "user_id: ${user_id}" \
        -H "x-token: ${token}" \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json'
}