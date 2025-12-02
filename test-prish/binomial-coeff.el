(defun binom-coeff (a b)
  (cond
    ((< a b) 0)
    ((= b 0) 1)
    ((= a b) 1)
    (t
     (let ((k (if (> b (- a b)) (- a b) b)))
       (/ (fact a) (* (fact k) (fact (- a k))))))))
       