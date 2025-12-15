# frontend/Dockerfile

# Use lightweight Node image
FROM node:18-alpine

# Set working directory inside the container
WORKDIR /app

# Copy package files first to optimize build cache
COPY package*.json ./

# Install dependencies
# '|| true' prevents crash if you haven't created package.json yet
RUN npm install || true

# Copy all frontend source files
COPY . .

# Expose the frontend port
EXPOSE 3000

# Start the Node.js server
CMD ["node", "server.js"]