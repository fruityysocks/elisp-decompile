(defun catalan (a)
  (if (eql a 0) 1
      (- (binom-coeff(* 2 a) a)
         (binom-coeff(* 2 a) (- a 1)))))
