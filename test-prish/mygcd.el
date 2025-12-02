(defun mygcd (a b)
  (cond
    ((eql b 0) a)
    ((eql a 0) b)
    ((> a b)
     (mygcd (- a b) b))
    (t
     (mygcd a (- b a)))))

