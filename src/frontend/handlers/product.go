package handlers

import (
	"log"
	"net/http"

	"github.com/gorilla/mux"
	"github.com/your-project/pb"
)

func (fe *frontendServer) getProduct(w http.ResponseWriter, r *http.Request) {
	id := mux.Vars(r)["id"]
	if id == "" {
		renderHTTPError(w, errors.New("product id not specified"), http.StatusBadRequest)
		return
	}
	log.Printf("[getProduct] getting product %s", id)
	p, err := fe.productCatalogSvcClient.GetProduct(r.Context(), &pb.GetProductRequest{Id: id})
	if err != nil {
		renderHTTPError(w, err, http.StatusInternalServerError)
		return
	}
	// ...
} 