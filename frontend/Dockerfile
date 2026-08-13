# STAGE 1: Build the React application
FROM node:20-slim AS build
WORKDIR /app

# Copy package.json and package-lock.json first.
# This order is chosen for layer caching: Docker will cache the npm install layer
# and reuse it in subsequent builds unless these files change, speeding up builds.
COPY package*.json ./
RUN npm install

# Copy the rest of the source code
COPY . .

# Accept a build argument for the backend URL, with a sensible local default
ARG VITE_API_BASE_URL=http://localhost:8000
# Set it as an environment variable so Vite picks it up during the build process
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

# Build the frontend (produces the /app/dist folder)
RUN npm run build

# STAGE 2: Serve the application with Nginx
FROM nginx:alpine

# Copy only the static files from the build stage to Nginx's default serving directory.
# This discards the entire Node.js environment and node_modules, keeping the final image tiny.
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Use Nginx's default entrypoint which already starts the server correctly.
# No need to override CMD.
