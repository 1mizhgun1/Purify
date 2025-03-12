package middleware

import (
	"fmt"
	"net/http"

	"purify/src/utils"
)

func RecoverMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		recoverLogger := utils.GetLoggerFromContext(r.Context())

		defer func() {
			if err := recover(); err != nil {
				recoverLogger.Error(fmt.Sprintf("panic recovered: %v", err))
				w.WriteHeader(http.StatusInternalServerError)
			}
		}()

		next.ServeHTTP(w, r)
	})
}
