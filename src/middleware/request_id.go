package middleware

import (
	"context"
	"log/slog"
	"net/http"
	"strconv"

	"github.com/gorilla/mux"
	"github.com/satori/uuid"
	"purify/src/config"
)

type response struct {
	http.ResponseWriter
	code int
}

func (resp *response) WriteHeader(code int) {
	resp.code = code
	resp.ResponseWriter.WriteHeader(code)
}

func CreateRequestIDMiddleware(logger *slog.Logger) mux.MiddlewareFunc {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			reqID := uuid.NewV4().String()
			reqIDLogger := logger.With(slog.String("x-request-id", reqID))

			r = r.WithContext(context.WithValue(r.Context(), config.LoggerContextKey, reqIDLogger))
			resp := response{ResponseWriter: w}
			resp.Header().Set("X-Request-ID", reqID)
			resp.Header().Set("Content-Type", "application/json; charset=utf-8")

			reqIDLogger.
				With(slog.String("method", r.Method)).
				With(slog.String("uri", r.URL.Path)).
				Info("input request")

			next.ServeHTTP(&resp, r)

			if resp.code == 0 {
				resp.code = http.StatusOK
			}

			reqIDLogger.
				With(slog.String("method", r.Method)).
				With(slog.String("uri", r.URL.Path)).
				With(slog.String("status", strconv.Itoa(resp.code))).
				Info("finished request")
		})
	}
}
