FROM node:20.2-bullseye-slim

RUN apt-get -y update && \
	apt-get -y --no-install-recommends install ca-certificates build-essential git curl nano zsh
RUN sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

WORKDIR /code

# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# install app dependencies
COPY package.json ./
COPY package-lock.json ./

RUN npm install --silent
RUN npm install react-scripts@3.4.1 -g --silent

# add app
COPY . ./

# CMD ["npm", "start"]
CMD ["bash"]