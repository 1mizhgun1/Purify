# --- Stage 1: Build binary ---
FROM golang:1.23.2-alpine AS builder

WORKDIR /purify/
COPY . .

RUN go clean --modcache
RUN CGO_ENABLED=0 GOOS=linux go build -mod=readonly -o ./.bin ./cmd/main/main.go

# --- Stage 2: Final image with rsvg-convert ---
FROM debian:bookworm-slim

# Устанавливаем rsvg-convert и зависимости
RUN apt-get update && \
    apt-get install -y librsvg2-bin ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/
COPY --from=builder /purify/ /app/

ENV TZ="Europe/Moscow"
EXPOSE 8080

ENTRYPOINT ["/app/.bin"]