ip=$(grep nameserver /etc/resolv.conf | awk '{print $2}'); git config --global http.proxy http://$ip:10809; git config --global https.proxy http://$ip:10809
