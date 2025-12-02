(defun sum-evens(list)
  (or list 
      (cond 
            ( 0)
            (t 
                (let (
                      (sum (sum-evens (cdr list))))
                    (if (evenp (car list))
                            (+ sum (car list))
                          sum))))))

;; problems with this: The or form returns the first truthy value it encounters. Since a non-empty list is truthy in Lisp, this immediately returns the list itself without evaluating the rest of the code. So if you call (sum-evens '(2 4 6)), it just returns (2 4 6) instead of 12.
