FROM node:20-alpine

WORKDIR /app

COPY package.json .
RUN npm install

COPY index.html main.js style.css ./

CMD ["npx", "live-server", "--host=0.0.0.0", "--port=3000", "--no-browser"]
