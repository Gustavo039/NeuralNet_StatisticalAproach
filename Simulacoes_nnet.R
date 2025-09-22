##REDE NÃO GENERALIZADA

# Inicialização da rede com pesos aleatórios
init_network = function(input_size, hidden_size, output_size) {
  list(
    W1 = matrix(runif(input_size * hidden_size, -0.5, 0.5), nrow = hidden_size),
    b1 = matrix(0, nrow = hidden_size, ncol = 1),
    W2 = matrix(runif(hidden_size * output_size, -0.5, 0.5), nrow = output_size),
    b2 = matrix(0, nrow = output_size, ncol = 1)
  )
}


# Função de ativação ReLU
relu = function(x) pmax(0, x)
relu_deriv = function(x) ifelse(x > 0, 1, 0)

# Forward e backward
forward_backward = function(network, x, y) {
  # Forward
  z1 = network$W1 %*% x + network$b1
  a1 = relu(z1)
  z2 = network$W2 %*% a1 + network$b2
  y_pred = z2

  # Erro e perda
  loss = mean((y_pred - y)^2)

  # Backward (gradientes)
  dL_dz2 = 2 * (y_pred - y)
  dL_dW2 = dL_dz2 %*% t(a1)
  dL_db2 = dL_dz2

  dL_da1 = t(network$W2) %*% dL_dz2
  dL_dz1 = dL_da1 * relu_deriv(z1)
  dL_dW1 = dL_dz1 %*% t(x)
  dL_db1 = dL_dz1

  list(
    loss = loss,
    gradients = list(W1 = dL_dW1, b1 = dL_db1, W2 = dL_dW2, b2 = dL_db2)
  )
}



# Treinamento
train_nn = function(X, Y, hidden_size = 5, epochs = 1000, lr = 0.01) {
  input_size = nrow(X)
  output_size = nrow(Y)
  network = init_network(input_size, hidden_size, output_size)

  for (epoch in 1:epochs) {
    total_loss = 0

    for (i in 1:ncol(X)) {
      x_i = X[, i, drop = FALSE]
      y_i = Y[, i, drop = FALSE]

      fb = forward_backward(network, x_i, y_i)
      total_loss = total_loss + fb$loss

      # Atualização dos pesos
      network$W1 = network$W1 - lr * fb$gradients$W1
      network$b1 = network$b1 - lr * fb$gradients$b1
      network$W2 = network$W2 - lr * fb$gradients$W2
      network$b2 = network$b2 - lr * fb$gradients$b2
    }

    if (epoch %% 100 == 0) {
      cat(sprintf("Epoch %d, Loss: %.4f\n", epoch, total_loss / ncol(X)))
    }
  }

  return(network)
}



# Dados simulados: função simples y = x1 + x2
set.seed(42)
X = matrix(runif(200), nrow = 2)  # 2 variáveis, 100 observações
Y = matrix(colSums(X), nrow = 1)  # soma de x1 + x2

Y = Y + rnorm(100)

# Treinar rede
net = train_nn(X, Y, hidden_size = 4, epochs = 1000, lr = 0.05)

# Fazer predição
predict_nn = function(network, x) {
  z1 = network$W1 %*% x + network$b1
  a1 = relu(z1)
  z2 = network$W2 %*% a1 + network$b2
  return(z2)
}

y_pred = 1
for(i in 1:ncol(X)){
  y_pred[i] = predict_nn(net, X[,i])
}


mse = mean(Y - y_pred)^2

# Testar uma nova entrada
new_x = matrix(c(0.3, 0.7), ncol = 1)
pred = predict_nn(net, new_x)
cat("Predição para (0.3, 0.7):", pred, "\n")


# Ajustar codigo da nnet não generalizada
# Criar codigo de nnet generalizad em numero de camadas ocultas
# Comparar com outros modelos de regressão linear (via mqo e via gd)
























##REDE GENERALIZADA
# Funções de ativação
relu = function(x) pmax(0, x)
relu_deriv = function(x) ifelse(x > 0, 1, 0)

# Inicialização da rede com múltiplas camadas ocultas
init_network = function(input_size, hidden_sizes, output_size) {
  layer_sizes = c(input_size, hidden_sizes, output_size)
  n_layers = length(layer_sizes) - 1

  network = list()
  for (i in 1:n_layers) {
    network[[paste0("W", i)]] = matrix(runif(layer_sizes[i + 1] * layer_sizes[i], -0.5, 0.5),
                                       nrow = layer_sizes[i + 1])
    network[[paste0("b", i)]] = matrix(0, nrow = layer_sizes[i + 1], ncol = 1)
  }
  return(network)
}

# Forward e backward para múltiplas camadas
forward_backward = function(network, x, y) {
  n_layers = length(network) / 2
  activations = list(x)
  zs = list()

  # Forward pass
  for (i in 1:n_layers) {
    z = network[[paste0("W", i)]] %*% activations[[i]] + network[[paste0("b", i)]]
    zs[[i]] = z
    if (i < n_layers) {
      activations[[i + 1]] = relu(z)
    } else {
      activations[[i + 1]] = z  # Saída sem ativação
    }
  }

  y_pred = activations[[n_layers + 1]]
  loss = mean((y_pred - y)^2)

  # Backward pass
  gradients = list()
  delta = 2 * (y_pred - y)

  for (i in n_layers:1) {
    gradients[[paste0("W", i)]] = delta %*% t(activations[[i]])
    gradients[[paste0("b", i)]] = delta

    if (i > 1) {
      da = t(network[[paste0("W", i)]]) %*% delta
      dz = da * relu_deriv(zs[[i - 1]])
      delta = dz
    }
  }

  list(loss = loss, gradients = gradients, activations = activations)
}

# Treinamento da rede
train_nn = function(X, Y, hidden_sizes = c(5, 5), epochs = 1000, lr = 0.01) {
  input_size = nrow(X)
  output_size = nrow(Y)
  network = init_network(input_size, hidden_sizes, output_size)

  for (epoch in 1:epochs) {
    total_loss = 0
    for (i in 1:ncol(X)) {
      x_i = X[, i, drop = FALSE]
      y_i = Y[, i, drop = FALSE]

      fb = forward_backward(network, x_i, y_i)
      total_loss = total_loss + fb$loss

      # Atualização dos pesos
      for (name in names(fb$gradients)) {
        network[[name]] = network[[name]] - lr * fb$gradients[[name]]
      }
    }

    if (epoch %% 100 == 0) {
      cat(sprintf("Epoch %d, Loss: %.4f\n", epoch, total_loss / ncol(X)))
    }
  }

  return(network)
}

# Predição com múltiplas camadas
predict_nn = function(network, x) {
  n_layers = length(network) / 2
  a = x
  for (i in 1:n_layers) {
    z = network[[paste0("W", i)]] %*% a + network[[paste0("b", i)]]
    if (i < n_layers) {
      a = relu(z)
    } else {
      a = z
    }
  }
  return(a)
}

# Dados simulados
set.seed(42)
X = matrix(runif(200), nrow = 2)  # 2 variáveis, 100 observações
Y = matrix(colSums(X), nrow = 1) + rnorm(100)

# Treinamento com 2 camadas ocultas de 4 e 3 neurônios
net = train_nn(X, Y, hidden_sizes = c(4, 3), epochs = 1000, lr = 0.009)

# Predição
pred = predict_nn(net, X[, 1, drop = FALSE])
