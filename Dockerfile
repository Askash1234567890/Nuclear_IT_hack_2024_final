# FROM nginx

# WORKDIR /hackaton

# COPY ./index.html /hackaton/index.html
# COPY ./about.html /hackaton/about.html
# COPY ./vendors /hackaton/vendors
# COPY ./jsme /hackaton/jsme
# COPY ./images /hackaton/images
# COPY ./fonts /hackaton/fonts
# COPY ./css /hackaton/css
# COPY ./js /hackaton/js

# EXPOSE 5000

FROM nginx:latest
COPY default.conf /etc/nginx/nginx.conf
EXPOSE 80
