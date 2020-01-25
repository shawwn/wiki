#!/bin/bash
source .env
set -ex
src="${1}"
shift 1
exec curl -fsSL --request POST --url https://api.imgur.com/3/image --header "authorization: Client-ID $IMGUR_CLIENTID" --header 'content-type: multipart/form-data;' -F "image=@${src}" "$@"
